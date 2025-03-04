import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from typing import List, Dict, Tuple, Set
from mtrfg.utils.sys_utils import load_json
from networkx import DiGraph, all_topological_sorts, from_edgelist
import numpy as np
import random
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class GraphDataset(Dataset):
    def __init__(self,
                 data: List[Dict[str, str]] = None,
                 tokenizer = None,
                 padding = True,
                 split = None,
                 shuffle = False,
                 ):
        self.tokenizer = tokenizer
        self.padding = padding
        if not self.padding:
            raise NotImplementedError('Padding must be enabled.')
        self.label_index_map = get_mappings(data)

        if shuffle:
            random.shuffle(data)
        self.data = self.preprocess_data(data)
        self.split = split


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


    @classmethod
    def from_path(cls, path, **kwargs):
        data = load_json(path)
        return cls(data, **kwargs)
    

    def augment(self, k = 1, keep_og = False):
        augmented_data = []
        for sample in tqdm(self.data, total=len(self.data), desc=f'Augmenting {self.split} dataset...'):
            sample_size = k
            if len(sample['step_graph']) > 0:
                G = from_edgelist(sample['step_graph'], create_using=DiGraph)
                all_topos = np.array(list(all_topological_sorts(G)))
                num_topos = all_topos.shape[0]
                if num_topos > 1:
                    all_topos = all_topos[1:]  # the 0th element is the base order, so we exclude it
                    if sample_size > len(all_topos):
                        sample_size = len(all_topos)
                    perms_mask = np.random.choice(len(all_topos), size = sample_size, replace=False)
                    perms = all_topos[perms_mask].tolist()
                    for i, perm in enumerate(perms):
                        perm_filled = torch.as_tensor(add_isolated_nodes(perm))
                        permuted_graph = self.permute_graph(sample, perm_filled)
                        augmented_data.append(permuted_graph)
                if keep_og:
                    augmented_data.append(sample)
                    
        print(f'Augmented {self.split} dataset from {len(self.data)} to {len(augmented_data)} samples.')
        # print(f"Augmented value counts: {dict(pd.DataFrame(augmented_data)['permuted'].value_counts())}")
        self.data = augmented_data
    
    def shuffle(self):
        random.shuffle(self.data)

    def only_use_max_step_graph(self):
        max = 0
        max_step_sample = None
        for sample in self.data:
            if len(sample['step_graph']) > 0:
                G = from_edgelist(sample['step_graph'], create_using=DiGraph)
                all_topos = list(all_topological_sorts(G))
                n_all_topos = len(all_topos)
                if n_all_topos > max:
                    max = n_all_topos
                    max_step_sample = sample
                    max_all_topos = all_topos
        self.data = [max_step_sample]

    
    def populate_texts(self):
        for sample in self.data:
            sample['updated_text']

    def permute_graph(self, G, order_idx):
        step_indices = G['step_indices']
        step_indices_tokens = G['step_indices_tokens']
        G_perm = G.copy()
        
        for key, value in G_perm.items():    
            idx = ('step_indices_tokens', step_indices_tokens) if ('tokens' in key or 'encoded_input' == key) else ('step_indices', step_indices)
            if is_tensorizable(value):
                G_perm[key] = apply_sub_dicts(value, lambda x: reorder_tensor(x, idx=idx[1], permutation=order_idx))
            elif key != 'step_graph':
                G_perm[key] = reorder_list(value, idx=idx[1], permutation=order_idx)
        return G_perm


    def preprocess_data(self, data):
        # turns all fields into appropriate tensors
        processed_data = []
        for sample in data:
            processed_sample = {}
            processed_sample.update(sample)
            encoding = self.tokenizer(sample['words'],
                                        is_split_into_words = True,
                                        return_tensors = 'pt',
                                        # padding = 'max_length' if self.padding else False,
                                        )
            word_ids = torch.as_tensor([elem if elem is not None else -100 for elem in encoding.word_ids()])
            words_mask_custom = torch.as_tensor([1 for _ in range(len(sample['words']))])
            encoding.update({'words_mask_custom': words_mask_custom, 'word_ids_custom': word_ids})
            processed_sample['encoded_input'] = encoding
            processed_sample['step_indices_tokens'] = self.convert_to_token_indices(sample['step_indices'], word_ids)
            processed_sample['pos_tags'] = torch.tensor([self.label_index_map['tag2class'][el] for el in sample['pos_tags']])
            processed_sample['pos_tags_tokens'] = self.convert_to_token_indices(processed_sample['pos_tags'], word_ids)
            processed_sample['head_tags'] = torch.tensor([self.label_index_map['edgelabel2class'][el] for el in sample['head_tags']])
            processed_sample['head_tags_tokens'] = self.convert_to_token_indices(processed_sample['head_tags'], word_ids)
            processed_sample['head_indices_tokens'] = self.convert_to_token_indices(processed_sample['head_indices'], word_ids)
            processed_sample['edge_index_full'] = torch.as_tensor([[head, tail] for head, tail in enumerate(sample['head_indices'])])
            processed_sample['edge_index_steps'] = torch.as_tensor([[head, tail] for head, tail in enumerate(sample['step_indices'])])
            processed_sample['edge_index_full_tokens'] = torch.as_tensor([[head, tail] for head, tail in enumerate(processed_sample['head_indices_tokens'])])
            processed_sample['edge_index_steps_tokens'] = torch.as_tensor([[head, tail] for head, tail in enumerate(processed_sample['step_indices_tokens'])])
            processed_sample = apply_sub_dicts(processed_sample, self.tensorize)
            processed_sample = apply_sub_dicts(processed_sample, self.pad)
            processed_sample['step_graph'] = self.get_step_graph(sample)
            processed_data.append(processed_sample)
        return processed_data


    def convert_to_token_indices(self, input: list, word_ids: torch.tensor):
        return torch.tensor([input[el] if el != -100 else 0 for el in word_ids], dtype=torch.long)


    def get_step_graph(self, sample):
        step_indices = torch.as_tensor(sample['step_indices'])
        head_indices = torch.as_tensor(sample['head_indices'])
        target_steps = torch.cat([torch.tensor([0]), step_indices])[head_indices]
        G_loops = torch.vstack([step_indices, target_steps])
        mask_steps = torch.where(G_loops[0] != G_loops[1], True, False) # filter out edges within the same step
        G_masked = G_loops[:, mask_steps]
        mask_zeros = torch.where(G_masked[1] != 0, True, False) # mask again to remove edges going to the R00T
        G_masked = G_masked[:, mask_zeros].T.tolist()
        G = [tuple(sorted([el[0], el[1]])) for el in G_masked]
        G = set(sorted(G, key=lambda x: x[0]))
        # G_s = torch.tensor([el[0] for el in G])
        # G_t = torch.tensor([el[1] for el in G])
        # G = torch.stack([G_s, G_t])
        return G
    

    def pad_zeros(self, t):
        if isinstance(t, torch.Tensor):
            t = t.squeeze()
            if len(t.shape) == 1:
                t = t.unsqueeze(1)
                padding_zeros = torch.zeros((self.tokenizer.model_max_length - t.shape[0], t.shape[1]), dtype=t.dtype)
                # use the same type for the 0s so the tensor doesn't change type
                t_padded = torch.cat([t, padding_zeros]).squeeze()
                return t_padded
            else:
                padding_zeros = torch.zeros((self.tokenizer.model_max_length - t.shape[0], t.shape[1]), dtype=t.dtype)
                t_padded = torch.cat([t, padding_zeros], dim = 0)
                return t_padded
        elif isinstance(t, list):
            len_padding = self.tokenizer.model_max_length - len(t)
            padding_zeros = [0] * len_padding
            type_internal = type(t[0])
            return t + [type_internal(el) for el in padding_zeros] 
    

    def tensorize(self, data):
        if is_tensorizable(data):
            return torch.as_tensor(data)
        else:
            return data


    def pad(self, t):
        if is_paddable(t) and self.padding:
            return self.pad_zeros(t)
        else:
            raise NotImplementedError('Padding must be enabled.')

def get_mappings(data):
    all_pos_tags = []
    all_head_tags = []

    for line in data:
        all_pos_tags += line['pos_tags']
        all_head_tags += line['head_tags']
    
    # Count frequency of each tag
    pos_tag_counts = {}
    for tag in all_pos_tags:
        pos_tag_counts[tag] = pos_tag_counts.get(tag, 0) + 1
    
    head_tag_counts = {}
    for tag in all_head_tags:
        head_tag_counts[tag] = head_tag_counts.get(tag, 0) + 1
    
    # Sort by frequency (highest to lowest)
    sorted_pos_tags = sorted(pos_tag_counts.items(), key=lambda item: item[1], reverse=True)
    sorted_head_tags = sorted(head_tag_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Create mappings (index 0 is reserved for 'no_label' in POS tags)
    pos_tags_map = {tag: i+1 for i, (tag, _) in enumerate(sorted_pos_tags)}
    pos_tags_map.update({'no_label': 0})
    
    # Head tags start at index 0 (no special reserved index)
    head_tags_map = {tag: i for i, (tag, _) in enumerate(sorted_head_tags)}
    
    # Return in the unified format
    return {'tag2class': pos_tags_map, 'edgelabel2class': head_tags_map}

def apply_sub_dicts(data, func):
    if hasattr(data, 'keys'):
        return {key: apply_sub_dicts(value, func) for key, value in data.items()}
    else:
        return func(data)

def reorder_tensor(t: torch.Tensor = None, idx = None, permutation = None):
    '''
    Produces a tensor where the elements are permuted based on an idx.
    Example input:
    permutation = torch.tensor([3, 1, 2, 4])
    idx = torch.tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Returns:
        >>> torch.tensor([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    '''
    idx_reordered = torch.arange(len(idx))
    idx_tensor = []
    for perm in permutation:
        perm_idx_tensor = torch.where(idx == perm)[0]
        idx_tensor += perm_idx_tensor
    left = min(idx_tensor)
    right = max(idx_tensor)
    idx_reordered[left:right+1] = torch.as_tensor(idx_tensor)
    t_reordered = t[idx_reordered]
    return t_reordered

def reorder_list(L: torch.Tensor = None, idx = None, permutation = None):
    '''
    Produces a tensor where the elements are permuted based on an idx.
    Example input:
    permutation = torch.tensor([3, 1, 2, 4])
    idx = torch.tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Returns:
        >>> torch.tensor([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    '''
    permutation = permutation.tolist()
    idx_original = list(range(len(idx)))
    idx_permuted = []
    for perm in permutation:
        perm_idx_tensor = [i for i, el in enumerate(idx) if el == perm]
        idx_permuted += perm_idx_tensor
    left = min(idx_permuted)
    right = max(idx_permuted)
    idx_original[left:right+1] = idx_permuted
    L_reordered = [L[i] for i in idx_original]
    return L_reordered

def adj_list_2_edge_index(L):
    edge_index = np.array([list(el) for el in L]).T
    return edge_index

def is_paddable(L):
    if isinstance(L, list):
        return True
    elif isinstance(L, torch.Tensor):
        return True
    elif isinstance(L, set):
        return False
    elif hasattr(L, 'keys'):
        return all([is_tensorizable(L[key]) for key in L.keys()])
    else:
        raise NotImplementedError('Not a list, tensor, set, or dict-like.')

def is_tensorizable(L):
    if isinstance(L, list):
        if len(L) == 0 or any([isinstance(el, str) for el in L]) or any([isinstance(el, set) for el in L]):
            return False
        else:
            return True
    elif isinstance(L, torch.Tensor):
        return True
    elif isinstance(L, set):
        return False
    elif hasattr(L, 'keys'):
        return all([is_tensorizable(L[key]) for key in L.keys()])
    else:
        raise NotImplementedError('Not a list, tensor, set, or dict-like.')

# def get_mappings(data):
#     all_pos_tags = []
#     all_head_tags = []

#     for line in data:
#         all_pos_tags+=line['pos_tags']
#         all_head_tags+=line['head_tags']
    
#     pos_tags_map = {k: i for i, k in enumerate(sorted(set(all_pos_tags)))}
#     head_tags_map = {k: i for i, k in enumerate(sorted(set(all_head_tags)))}

#     return pos_tags_map, head_tags_map

def add_isolated_nodes(L):
    min_val = min(L)
    max_val = max(L)

    # Find the missing numbers in the range
    full_range = set(range(min_val, max_val + 1))
    missing_values = list(full_range - set(L))

    # Insert each missing value at a random position
    for value in missing_values:
        random_index = random.randint(0, len(L))
        L.insert(random_index, value)

    return L

class GraphCollator:
    def __init__(self, keys = ['words', 'step_graph'], truncate_to_longest = True):
        self.keys = keys
        self.truncate_to_longest = truncate_to_longest
    
    def collate(self, input):
        out, filtered = self.filter_keys(input)
        if self.truncate_to_longest:
            max = self.get_trunc_len(input)
            out = self.truncate(out, max)
        out = default_collate(out)
        for key in self.keys:
            out[key] = [el[key] for el in filtered]
        return out
    
    def truncate(self, batch, max):
        for i in range(len(batch)):
            for key, value in batch[i].items():
                if isinstance(value, torch.Tensor):
                    batch[i][key] = batch[i][key][:max]
            for key, value in batch[i]['encoded_input'].items():
                batch[i]['encoded_input'][key] = batch[i]['encoded_input'][key][:max]
        return batch


    def get_trunc_len(self, batch):
        max = 0
        for el in batch:
            input_ids = el['encoded_input']['input_ids']
            new = len(input_ids[torch.where(input_ids != 0)])
            if new > max:
                max = new
        return max

    def filter_keys(self, input):
        batch = []
        filtered = []
        for el in input:
            out_el = {}
            filtered_el = {}
            for key, value in el.items():
                if key not in self.keys:
                    out_el[key] = value
                else:
                    filtered_el[key] = value
            batch.append(out_el)
            filtered.append(filtered_el)
        return batch, filtered

def transformer_input_filter(input, keys = ['input_ids', 'token_type_ids', 'attention_mask']):
    return {key: input[key] for key in keys}

def build_dataloaders(dataset_dict, collate_fn, batch_size = 1, splits = ['train', 'val', 'test']):
    return {
        split: DataLoader(dataset_dict[split],
                   batch_size=batch_size,
                   collate_fn=collate_fn
                   ) for split in splits
        }

def save_heatmap(matrix, filename="heatmap.pdf", cmap="viridis"):
    """
    Saves a heatmap of a square matrix as a PDF.

    Parameters:
    - matrix (torch.Tensor or np.ndarray): Square matrix to visualize.
    - filename (str): Output filename (default: "heatmap.pdf").
    - cmap (str): Matplotlib colormap (default: "viridis").
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()  # Convert PyTorch tensor to NumPy

    # if matrix.shape[0] != matrix.shape[1]:
    #     raise ValueError("Input matrix must be square.")

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.title("Heatmap")
    
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def show_attentions(model_out):
    # Extract attention matrices (Shape: num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = model_out.attentions

    # Get attention maps from the last layer
    last_layer_attentions = attentions[-1][0]  # Shape: (num_heads, seq_len, seq_len)
    num_heads = last_layer_attentions.shape[0]

    # Plot heatmaps for each attention head in the last layer
    fig, ax = plt.subplots(figsize=(20, 15))  # 12 heads (4 rows × 3 cols)

    sns.heatmap(last_layer_attentions[0].detach().cpu().numpy(), ax=ax, cmap="Blues")

    plt.tight_layout()
    plt.savefig('model_attentions.pdf', format='pdf')

def main():
    pass

if __name__ == "__main__":
    main()

