custom_config = {
    'train_file_graphs': 'data/train.json', ## should be supplied at the time of command. --opts --train_file <path/to/train/file>
    'val_file_graphs': 'data/val.json',
    'test_file_graphs': 'data/test.json',
    
    'test_ignore_edge_dep': ['root', '-'], ## this will be ignored during evaluation

    'shuffle': {'train': 0, 'val': 0, 'test': 0},

    'only_use_biggest_graph': {'train': 0, 'val': 0, 'test': 0},
    'biggest_graph_threshold': 0,
    'augment_splits': {'train': 0, 'val': 0, 'test': 0},  # 1s or 0s
    'augment_k': {'train': 0, 'val': 0, 'test': 0,},
    'keep_og': 0,
    'rep_mode': 'words', # either 'words' or 'tokens'. right now 'tokens' doesn't work in the evaluation because the correct labels need to be used o.w. the lenghts don't match.
    'use_bert_positional_embeddings': 1,
    'use_tag_embeddings_in_parser': 0,  # `True` is the default MTRFG setting
    'use_tagger_lstm': 0, # `True` is the default MTRFG setting
    'use_parser_lstm': 1,  # `True` is the default MTRFG setting
    'use_gnn': 0,
    'use_step_mask': 0,

    'freeze_encoder': 1,
    'learning_rate': 1e-3,
    'batch_size': 8,

    'save_model': 1,

    'data_len': {'train': 0, 'val': 0, 'test': 0,},
    'parser_lstm_hidden_size': 400,
}

default_cfg = {
    'device': 'cuda:0', ## will be modified when we get actual device using torch
    'optimizer': 'dense_sparse_adam', ## supported adam, sparse_adam, dense_sparse_adam, huggingface_adamw, sgd
    'train_file': 'data/Parser/train.conllu', ## should be supplied at the time of command. --opts --train_file <path/to/train/file>
    'val_file': 'data/Parser/dev.conllu',
    'test_file': 'data/Parser/test.conllu',
    'freeze_encoder': 1,
    'freeze_tagger': 0,
    'freeze_parser': 0,
    'early_stopping': True,
    'patience': 30,  ## patience period for early stopping
    'epochs': 100,
    'learning_rate': 0.001,
    'shuffle' : True, 
    'save_dir': './saved_models',
    'model_name': 'bert-base-uncased', ## model name, should be key in hugging face pretrained model zoo
    'batch_size': 8,
    'encoder_output_dim': 768,
    'sparse_embedding_tags' : False,
    'n_tags': None, ## to be calculated after loading train data
    'n_edge_labels': None, ## to be calculated after loading train data
    'freeze_until_epoch' : 999, ## it freezes the encoder until a given epoch number, then unfreezes it. If freeze_until_epoch > epochs, then entire training will be with frozen encoder
    'test_ignore_tag': ['O', 'no_label'], ## this will be ignored during evaluation
    'test_ignore_edge_dep': ['root'], ## this will be ignored during evaluation
    'tag_embedding_dimension': 100,
    'seed': 27,
    'self_attention_heads': 2,
    'use_multihead_attention' : False, ## whether to use multihead attention in encoder or not! This is a learnable module that would help in generating better representations.
    'fraction_dataset': 1, ## this is between 0 and 1, if it's 0.2, then only 20% of train, test and validation dataset will be used. This is used to do hyperparameter search for training where we do training on subset of the dataset
    'betas': [0.9, 0.9],
    'use_pred_tags' : True, ## this will determine if gold tags are used for train/test/validation or not. 
    'keep_edge_labels': True,
    'softmax_scaling_coeff' : 1000,
    'gumbel_softmax' :  False, ## using gumbel softmax in tagger, it's false by default.
    'keep_tags':True, 
    'model_path' : None,
}