import os
import torch

from torch.utils.data import Dataset, DataLoader

import datasets

from transformers import AutoTokenizer

from data.utils import clean_data

class FixedLengthDataset(Dataset):
    def __init__(self, orig_dataset, length):
        self.orig_dataset = orig_dataset
        self.data = {}
        self.length = length

    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):

        if idx not in self.data:

            orig_item = self.orig_dataset[idx]

            item = {}
            item['labels'] = orig_item['labels']

            input_ids = torch.zeros(self.length, dtype=orig_item['input_ids'].dtype, device = orig_item['input_ids'].device)
            input_ids[:orig_item['input_ids'].shape[0]] = orig_item['input_ids']
            item['input_ids'] = input_ids

            attention_mask = torch.zeros(self.length, dtype=orig_item['attention_mask'].dtype, device = orig_item['attention_mask'].device)
            attention_mask[:orig_item['attention_mask'].shape[0]] = orig_item['attention_mask']
            item['attention_mask'] = attention_mask

            self.data[idx] = item

        return self.data[idx]
    
def tokenize_datasets(model_name, dataset):

    tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=2)
    #tokenizer.pad_token = tokenizer.eos_token 

    def tokenize_function(examples):
        try:
            return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")
        except:
            tokenizer.pad_token = tokenizer.eos_token 
            return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")

    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column('label', "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets

def get_dataloader(model_name, data_path, use_cache=True, batch_size=32, shuffle=True, dataset_name='new_all_DEFR_comments_27062022.csv', debug=0, model_nickname='', label_column='hatespeech', comment_column='kommentar'):

    cache_path = data_path + 'cache/'
    cache_file = f'{cache_path}{model_nickname}_{label_column}_{dataset_name[:-4]}_cached_dataset.pt'
    data_file = data_path + dataset_name

    if use_cache and os.path.exists(cache_file):
        tokenized_dataset = torch.load(cache_file)
    else:
        # Create cache dir if it does not exist
        if not os.path.exists(cache_path): os.makedirs(cache_path)
    
        df_processed = clean_data(data_file=data_file, label_column=label_column, comment_column=comment_column)
        dataset = datasets.Dataset.from_pandas(df_processed, preserve_index=False)

        if debug > 0:
            return dataset
        
        tokenized_dataset = tokenize_datasets(model_name=model_name, dataset=dataset)
        if use_cache: torch.save(tokenized_dataset, cache_file)

    max_length = max([x.shape for x in tokenized_dataset['input_ids']])

    dataset_fl = FixedLengthDataset(tokenized_dataset, length=max_length)

    data_loader = DataLoader(dataset_fl, shuffle=shuffle, batch_size=batch_size, num_workers=32)

    return data_loader


