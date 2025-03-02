from torch.utils.data import DataLoader
from mtrfg.utils.graph_data_utils import GraphCollator, GraphDataset
from transformers import AutoTokenizer

def build_dataloader(config, loader_type = 'train', augment_k = 1, only_permuted = True):
    collator = GraphCollator()
    kwargs = {'add_prefix_space' : True}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    
    if loader_type == 'train':
        train_data = GraphDataset.from_path(config['train_file_graphs'], tokenizer = tokenizer, split = 'train')
        if loader_type in config['augment_splits']:
            train_data.augment(k = augment_k, only_permuted=only_permuted)
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], collate_fn = collator.collate, shuffle = False)
        return train_loader

    if loader_type == 'val':
        val_data = GraphDataset.from_path(config['val_file_graphs'], tokenizer = tokenizer, split = 'val')
        if loader_type in config['augment_splits']:
            train_data.augment(k = augment_k, only_permuted=only_permuted)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn = collator.collate)
        return val_loader

    if loader_type == 'test':
        test_data = GraphDataset.from_path(config['test_file_graphs'], tokenizer = tokenizer, split = 'test')
        if loader_type in config['augment_splits']:
            train_data.augment(k = augment_k, only_permuted=only_permuted)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], collate_fn = collator.collate)
        return test_loader

    return None