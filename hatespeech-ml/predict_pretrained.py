import torch
import itertools


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data.dataset import get_dataloaders
from train.sequence import ClassificationWrapper

from pytorch_lightning.callbacks import ModelCheckpoint

import torch_optimizer as optim

from train.deepspeed_sequence import TextClassificationTransformer

num_classes = 2

model_name = "intfloat/multilingual-e5-large"
#model_name = "unitary/multilingual-toxic-xlm-roberta"

data_path = "../data/"

class_weight = torch.tensor([1.0, 4.6]) #torch.tensor([1.0, 2*4.60967657991111])
learning_rate = 1e-6
batch_size = 16
epochs=3

optimizer = opti_params = None
callbacks = []

num_classes = 2

def main():
        
    #optimizer = optim.Adafactor
    #opti_params = {'lr' : 1e-3, 'eps2' : (1e-30, 1e-3), 'clip_threshold' : 1.0, 'decay_rate' : -0.8, 'beta1' : 0.0, 'weight_decay': 0.0, 'scale_parameter' : True, 'relative_step' : True, 'warmup_init' : False}
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        
        model = TextClassificationTransformer(model_name, num_labels=num_classes, ignore_mismatched_sizes=True, cache_dir="../data/huggingface", 
                                                deepspeed_sharding=False, loss=criterion, learning_rate=learning_rate, weight_decay=0.01)

        trainer = pl.Trainer(accelerator="gpu", strategy='deepspeed', devices='auto', precision='16-mixed', logger=False)

        train_loader, eval_loader = get_dataloaders(model_name=model_name, data_path=data_path, dataset_name='german.csv',
                                                    batch_size=batch_size, use_cache=True, nickname='german_bert_existing', label_column='toxic_new')
        
        trainer.test(model, dataloaders=eval_loader)

if __name__ == '__main__':
    main()