import torch
from mtrfg.encoder import Encoder
from mtrfg.parser import BiaffineDependencyParser
from mtrfg.tagger import Tagger
import numpy as np
import warnings
from typing import Set, Tuple

class MTRfg(torch.nn.Module):
    def __init__(self, config): 
        super(MTRfg, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.tagger = Tagger(config)
        self.parser = BiaffineDependencyParser.get_model(config)
        self.mode = 'train'

    def freeze_tagger(self):
        """ Freeze tagger if asked for!"""
        for param in self.tagger.parameters():
            param.requires_grad = False

    def freeze_parser(self):
        """ Freeze parser if asked for!"""
        for param in self.parser.parameters():
            param.requires_grad = False

    def get_output_as_list_of_dicts(self, tagger_ouptut, parser_output, model_input):
        """
            Returns list of dictionaries, each element in the dictionary is 
            1 item in the batch, list has same length as batchsize. The dictionary
            will contain 7 fields, 'words', 'head_tags_gt', 'head_tags_pred', 'pos_tags_gt',
            'pos_tags_pred', 'head_indices_gt', 'head_indices_pred'. During evalution, all fields
            should have exactly identical length, during testing, '*_gt' keys() will have empty 
            tensors.
        """
        outputs = []
        batch_size = len(tagger_ouptut)

        for i in range(batch_size):
            elem_dict = {}
            
            ## find non-masked indices
            valid_input_indices = torch.where(model_input['encoded_input']['words_mask_custom'][i] == 1)[0].cpu().detach().numpy().tolist()

            input_length = len(valid_input_indices)

            elem_dict['words'] = np.array(model_input['words'][i])[valid_input_indices].tolist()
            
            elem_dict['head_tags_gt'] = model_input['head_tags'][i].cpu().detach().numpy()[valid_input_indices].tolist()
            elem_dict['head_tags_pred'] = parser_output['predicted_dependencies'][i]

            elem_dict['head_indices_gt'] = model_input['head_indices'][i].cpu().detach().numpy()[valid_input_indices].tolist()
            elem_dict['head_indices_pred'] = parser_output['predicted_heads'][i]
            
            elem_dict['pos_tags_gt'] = model_input['pos_tags'][i].cpu().detach().numpy()[valid_input_indices].tolist()
            elem_dict['pos_tags_pred'] = tagger_ouptut[i]
            
            assert np.all([len(elem_dict[key]) == input_length for key in elem_dict]), "Predictions are not same length as input!"
            ## append
            outputs.append(elem_dict)

        return outputs

    def set_mode(self, mode = 'train'):
        """
            This function will determine if loss should be computed or evaluation metrics
        """
        assert mode in ['train', 'test', 'validation'], f"Mode {mode} is not valid. Mode should be among ['train', 'test', 'validation'] "
        self.tagger.set_mode(mode)
        self.mode = mode


    def forward(self, model_input):
        model_input = {k: v.to(self.config['device']) if isinstance(v, torch.Tensor) else v for k, v in model_input.items()}
        ## Building representations
        encoder_input = {k: v.to(self.config['device']) if isinstance(v, torch.Tensor) else v for k, v in model_input['encoded_input'].items()}
        original_mask = encoder_input['attention_mask']
        # If we use a step mask, its token-wise version needs to go into BERT
        # but after that there's no need to care about the step mask, just use `words_mask_custom`.
        if self.config['use_step_mask']:
            encoder_input['attention_mask'] = self.make_step_mask(original_mask,
                                                                  model_input['step_graph'],
                                                                  model_input['step_indices_tokens'])

        # The token-wise normal mask/step mask goes into BERT here.
        # In the encoder below and in the next if statement
        # we control whether we are going to use word-wise or token-wise representations.
        encoder_output = self.encoder(encoder_input) ## We get new attention mask because we have merged representations.

        
        if self.config['rep_mode'] == 'words':
            tagger_labels = model_input['pos_tags']
            downstream_mask = encoder_input['words_mask_custom']
            head_indices = model_input['head_indices']
            head_tags = model_input['head_tags']
        elif self.config['rep_mode'] == 'tokens':
            tagger_labels = model_input['pos_tags_tokens']
            downstream_mask = original_mask
            head_indices = model_input['head_indices_tokens']
            head_tags = model_input['head_tags_tokens']

        ## tagging the input
        tagger_output = self.tagger(encoder_output, mask = downstream_mask, labels = tagger_labels)

        ## predicted tags
        pos_tags_pred = self.tagger.get_predicted_classes_as_one_hot(tagger_output.logits)

        ## tags
        try:
            pos_tags_gt = torch.nn.functional.one_hot(tagger_labels, num_classes = self.config['n_tags'])
        except:
            warnings.warn("Ground truth tags are unavailable, using predicted tags for all purposes.")
            pos_tags_gt = pos_tags_pred

        ## during training, we use gt labels, otherwise, we use predicted labels
        if self.mode in ['train', 'validation']:  
            head_tags, head_indices = head_tags, head_indices
        
        elif self.mode == 'test':
            # pos_tags = tagger_labels
            head_tags, head_indices = None, None
        
        pos_tags_parser = pos_tags_pred if self.config['use_pred_tags'] else pos_tags_gt
        parser_output = self.parser(encoder_output, pos_tags_parser.float(), downstream_mask, head_tags = head_tags, head_indices = head_indices)

        ## calculate loss, when training or validation
        if self.mode in ['train', 'validation']:
            loss = 25 * (parser_output['loss'] + tagger_output.loss)
            return loss
        elif self.mode == 'test':
            tagger_human_readable = self.tagger.make_output_human_readable(tagger_output, downstream_mask)
            parser_human_readable = self.parser.make_output_human_readable(parser_output)            
            output_as_list_of_dicts = self.get_output_as_list_of_dicts(tagger_human_readable, parser_human_readable, model_input)
            return output_as_list_of_dicts
            ## when not training
            ## get human readable outputs

    def make_step_mask(self, attention_mask: torch.Tensor, step_graph: Set[Tuple[int, int]], step_idx_tokens: torch.Tensor):
        '''
        This function takes in an attention mask, a step graph, and per-token step indices
        to return a mask where the ones (1) are only present in areas
        relative to nodes connected within the step graph
        '''
        step_idx_tokens = step_idx_tokens.to(self.config['device'])
        
        _, seq_len = attention_mask.shape
        step_mask_list = []

        for nodes, step_indices, attn_mask in zip(step_graph, step_idx_tokens, attention_mask):
            step_graph_rev = set([tuple(sorted(el, reverse=True)) for el in nodes])
            nodes = nodes.union(step_graph_rev)
            unique_nodes = set(step_indices.tolist()).difference({0})

            # Initialize mask with zeros
            mask = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=self.config['device'])

            # Add self-loops
            for i in unique_nodes:
                mask += ((step_indices[:, None] == i) & (step_indices[None, :] == i)).int()

            # Add edges
            for src, tgt in nodes:
                mask += ((step_indices[:, None] == src) & (step_indices[None, :] == tgt)).int()

            pad_limit = torch.max(torch.where(attn_mask == 1)[0])

            # NOTE: these two lines include the SEP tokens, don't remove
            mask[:pad_limit+1,pad_limit:pad_limit+1] = 1
            mask[pad_limit:pad_limit+1,:pad_limit+1] = 1
            # NOTE: this line includes the original padding, don't remove
            mask[pad_limit:,:pad_limit+1] = 1

            # NOTE: the line below is the equivalent of using the original mask
            # mask[:,:pad_limit+1] = 1

            step_mask_list.append(mask)
        
        # save_heatmap(mask, filename='step_mask1.pdf')
        
        batch_step_mask = torch.stack(step_mask_list, dim=0)
        return batch_step_mask