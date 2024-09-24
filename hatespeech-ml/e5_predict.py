import torch
import itertools


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data.binary_nli_dataset import get_dataloader

from pytorch_lightning.callbacks import ModelCheckpoint

from train.deepspeed_sequence import TextClassificationTransformer
from data.multi_gpu_pred_writer import CustomWriter

num_classes = 2

model_name = "intfloat/multilingual-e5-large"

data_path = "../data/"

class_weight = torch.tensor([1.0, 4.614545638159998])
#class_weight = torch.tensor([1.0, 3.2790649942987455])
learning_rate = 1e-5
batch_size = 16
epochs=10

optimizer = opti_params = None
callbacks = []

num_classes = 2

test_loader = get_dataloader(
    model_name=model_name, 
    data_path=data_path, 
    dataset_name='oct23_dummy.csv',
    batch_size=batch_size, 
    use_cache=True, 
    model_nickname='e5', 
    comment_column='body',
    label_column='dummy',
    shuffle=False,
)

writer = CustomWriter(
    write_interval='epoch',
    output_file=data_path + 'outputs/e5_oct23_preds.pkl'
)

callbacks = [writer]

trainer = pl.Trainer(
    accelerator="gpu", 
    strategy='deepspeed', 
    devices='auto', 
    precision='16',#'16-mixed', 
    logger=False, 
    max_epochs=epochs, 
    callbacks=callbacks, 
    log_every_n_steps=50
)

from pathlib import Path

def main():
        
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    
    model = TextClassificationTransformer(
        model_name, 
        num_labels=num_classes, 
        ignore_mismatched_sizes=True, 
        cache_dir="../data/huggingface", 
        deepspeed_sharding=True, 
        loss=criterion, 
        learning_rate=learning_rate, 
        weight_decay=0.01, 
    )
    ckpt_path = Path("/home/rasteiger/datasets/dslab/checkpoint/e5_nli/adamw/cw_1/bs_16/seed_43/model-epoch=02-train/F1Score=0.74.ckpt")
    #trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    #trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)
    trainer.predict(model, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()