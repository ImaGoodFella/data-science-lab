import torch
import itertools

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data.nli_dataset import get_dataloader

from pytorch_lightning.callbacks import ModelCheckpoint

from train.deepspeed_sequence import TextClassificationTransformer
from data.multi_gpu_pred_writer import CustomWriter

num_classes = 2

model_name = "mjwong/multilingual-e5-large-xnli"
data_path = "../data/"

learning_rate = 2e-6
batch_size = 16
epochs=3

optimizer = opti_params = None
callbacks = []

num_classes = 2

train_loader = get_dataloader(
    model_name=model_name, 
    data_path=data_path, 
    dataset_name='trainval.csv',
    batch_size=batch_size, 
    use_cache=True, 
    nickname='nli', 
    comment_column='kommentar_original',
    label_column='hatespeech',
    shuffle=True,
    is_test=True
)

test_loader = get_dataloader(
    model_name=model_name, 
    data_path=data_path, 
    dataset_name='test.csv',
    batch_size=batch_size, 
    use_cache=True, 
    nickname='nli', 
    comment_column='kommentar_original',
    label_column='hatespeech',
    shuffle=False,
    is_test=True
)

logger = pl_loggers.TensorBoardLogger(
    save_dir=data_path, name='logs/e5-large/', 
    version=f"nli_no_val_16"
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="train/F1Score",
    mode="max",
    dirpath=data_path +"logs/nli-large/",
    filename="nli_train_16-{epoch:02d}-{train/F1Score:.2f}",
    save_weights_only=False,
)

test_writer = CustomWriter(
    write_interval='epoch',
    output_file=data_path + 'outputs/nli_test_preds_16.pkl'
)

expert_writer = CustomWriter(
    write_interval='epoch',
    output_file=data_path + 'outputs/nli_expert_preds_16.pkl'
)

callbacks = [checkpoint_callback, test_writer]

trainer = pl.Trainer(
    accelerator="gpu", 
    strategy='deepspeed', 
    devices='auto', 
    precision='16',#'16-mixed', 
    logger=logger, 
    max_epochs=epochs, 
    callbacks=callbacks, 
    log_every_n_steps=50
)

def main():
        
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    criterion = torch.nn.CrossEntropyLoss()
    
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
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader)
    trainer.predict(model, dataloaders=test_loader)

if __name__ == '__main__':
    main()