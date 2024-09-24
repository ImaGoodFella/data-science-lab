import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import datasets

from transformers import AutoTokenizer

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

def prepare_data(df, is_test = False, comment_column='kommentar_original'):

    labels_en = np.array(["gender", "age", "sexuality", "religion", "nationality",
             "disability", "social status", "politics", "appearance", "others"])
    
    labels_de = np.array(['geschlecht', 'alter', 'sexualitaet', 'religion', 'nationalitaet',
       'beeintraechtigung', 'sozialer_status', 'politik', 'aussehen', 'andere'])

    hypothesis_base = "This text targets the user based on "
    premises, hypotheses, entail_labels, true_labels = [], [], [], []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        label_idx = np.where(row[labels_de])[0]

        labels = labels_en[label_idx]
        not_labels = list(set(labels_en) - set(labels))

        if not is_test:
            for label in labels:
                try:
                    neg_label = np.random.choice(not_labels)
                except:
                    continue
                #True label
                hypotheses.append(hypothesis_base + label)
                entail_labels.append(0)
                true_labels.append(labels)
                premises.append(row[comment_column])

                #Negative sampling
                hypotheses.append(hypothesis_base + neg_label)
                entail_labels.append(1)
                true_labels.append(labels)
                premises.append(row[comment_column])
        else:
            for label in labels_en:
                if label in labels:
                    entail_labels.append(0)
                else:
                    entail_labels.append(1)

                hypotheses.append(hypothesis_base + label)
                true_labels.append(labels)
                premises.append(row[comment_column])

    return pd.DataFrame(
        data = {
            'premise': premises,
            'hypothesis': hypotheses,
             'label': entail_labels,
             'true_label': true_labels
            }
        )

def tokenize_datasets(model_name, dataset):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer.pad_token = tokenizer.eos_token 

    def tokenize_function(examples):
        try:
            return tokenizer(examples['premise'], examples['hypothesis'], padding='max_length', truncation=True, return_tensors="pt")
        except:
            tokenizer.pad_token = tokenizer.eos_token 
            return tokenizer(examples['premise'], examples['hypothesis'], padding='max_length', truncation=True, return_tensors="pt")
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis"])
    tokenized_datasets = tokenized_datasets.rename_column('label', "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets

def get_dataloader(model_name, data_path, use_cache=True, batch_size=32, shuffle=True, dataset_name='new_all_DEFR_comments_27062022.csv', model_nickname='', is_test=False, comment_column='kommentar', label_column='hatespeech'):

    cache_path = data_path + 'cache/'
    cache_file = f'{cache_path}{model_nickname}_{label_column}_{dataset_name[:-4]}_pure_nli_cached_dataset.pt'
    data_file = data_path + dataset_name

    if use_cache and os.path.exists(cache_file):
        tokenized_dataset = torch.load(cache_file)
    else:
        # Create cache dir if it does not exist
        if not os.path.exists(cache_path): os.makedirs(cache_path)

        df = pd.read_csv(data_file)
        df_filtered = df[df[label_column].astype(bool)]
        df_processed = prepare_data(df_filtered, is_test, comment_column=comment_column)
        dataset = datasets.Dataset.from_pandas(df_processed, preserve_index=False)
        
        tokenized_dataset = tokenize_datasets(model_name=model_name, dataset=dataset)
        if use_cache: torch.save(tokenized_dataset, cache_file)

    max_length = max([x.shape for x in tokenized_dataset['input_ids']])

    dataset_fl = FixedLengthDataset(tokenized_dataset, length=max_length)
    data_loader = DataLoader(dataset_fl, shuffle=shuffle, batch_size=batch_size, num_workers=32)

    return data_loader