"""
    This file contains different encoders to be used for 
    extracting encoded representations from a transformer. 
"""


import torch 
import torch.nn as nn
from transformers import AutoModel
from mtrfg.encoder.bert_nopos import BertModelNoPos
from transformers import BatchEncoding
from typing import List, Dict, Optional
import inspect

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        if self.config['use_bert_positional_embeddings']:
            self.encoder = AutoModel.from_pretrained(self.config['model_name'])
        else:
            self.encoder = BertModelNoPos.from_pretrained(self.config['model_name'])
        self.encoder_input_keys = [key for key in inspect.signature(self.encoder.forward).parameters.keys()]

        if self.config['use_multihead_attention']:
            self.mha = nn.MultiheadAttention(self.config['encoder_output_dim'], self.config['self_attention_heads'], batch_first = True, dropout = 0.2)

    def forward(
        self, 
        encoded_input: Dict) -> torch.Tensor:

        input_to_encoder = {key : encoded_input[key] if key in encoded_input else None for key in self.encoder_input_keys}
        input_to_encoder = {key : value for key, value in input_to_encoder.items() if not key.endswith('_custom')}

        outputs = self.encoder(**input_to_encoder)
        
        if self.config['rep_mode'] == 'words':
            encoded_output = self.merge_subword_representation(outputs, encoded_input)
            attention_mask = encoded_input['words_mask_custom']
        elif self.config['rep_mode'] == 'tokens':
            encoded_output = outputs.last_hidden_state
            attention_mask = encoded_input['attention_mask']

        if self.config['use_multihead_attention']:
            print('Using new MHA after encoder output.')
            # `attention_mask` here was just the original encoded_input['attention_mask'] although the representation was merged
            encoded_output, _ = self.mha(encoded_output, encoded_output, encoded_output, key_padding_mask = attention_mask < 0.5 )
            
        return encoded_output

    def freeze_encoder(self):
        """
            freeze model parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
            unfreeze model parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

    def merge_subword_representation(self, outputs, encoded_input: BatchEncoding):
        """
            Merges subword representations
            Parameters:
            ----------
            outputs: Of type torch.Tensor
            encoded_input: Of type BatchEncoding
        """

        ## keep it same dimensional as original output to keep 
        ## it batch friendly!
        
        outputs_new = outputs.last_hidden_state.clone()

        for i, batch in enumerate(outputs.last_hidden_state):
            word_idxs = encoded_input['word_ids_custom'][i]

            tot_words = torch.max(word_idxs).item() + 1
            
            for word_idx in range(tot_words):
                """
                    merging subword representations
                """
                word_representation = batch[torch.where(word_idxs == word_idx)[0]].mean(dim = 0)
                outputs_new[i][word_idx] = word_representation
        
        return outputs_new
