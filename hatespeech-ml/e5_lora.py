import torch
import itertools

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data.dataset import get_dataloader

from pytorch_lightning.callbacks import ModelCheckpoint

from train.deepspeed_sequence import TextClassificationTransformer
from data.multi_gpu_pred_writer import CustomWriter

from peft import LoraConfig, TaskType

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
learning_rate = 3e-4

num_classes = 2

model_name = "intfloat/multilingual-e5-large"

data_path = "../data/"

class_weight = torch.tensor([1.0, 4.614545638159998])
#learning_rate = 5e-6
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
    model_nickname='e5', 
    comment_column='kommentar_original',
    label_column='hatespeech',
    shuffle=True,
)

test_loader = get_dataloader(
    model_name=model_name, 
    data_path=data_path, 
    dataset_name='test.csv',
    batch_size=batch_size, 
    use_cache=True, 
    model_nickname='e5', 
    comment_column='kommentar_original',
    label_column='hatespeech',
    shuffle=False,
)

expert_loader = get_dataloader(
    model_name=model_name, 
    data_path=data_path, 
    dataset_name='expert.csv',
    batch_size=batch_size, 
    use_cache=True, 
    model_nickname='e5', 
    comment_column='kommentar_original',
    label_column='hatespeech',
    shuffle=False,
)

logger = pl_loggers.TensorBoardLogger(
    save_dir=data_path, name='logs/e5_lora-large/', 
    version=f"e5_lora_no_val_{batch_size}"
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="train/F1Score",
    mode="max",
    dirpath=data_path +"logs/e5_lora-large/",
    filename=f"e5_lora_train_{batch_size}"+"-{epoch:02d}-{train/F1Score:.2f}",
    save_weights_only=False,
)

test_writer = CustomWriter(
    write_interval='epoch',
    output_file=data_path + f'outputs/e5_lora_test_preds_{batch_size}.pkl'
)

expert_writer = CustomWriter(
    write_interval='epoch',
    output_file=data_path + f'outputs/e5_lora_expert_preds_{batch_size}.pkl'
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

expert_trainer = pl.Trainer(
    accelerator="gpu", 
    strategy='deepspeed', 
    devices='auto', 
    precision='16',#'16-mixed', 
    logger=logger, 
    max_epochs=epochs, 
    callbacks=[expert_writer], 
    log_every_n_steps=50
)

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
        peft_config=peft_config
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader)
    trainer.predict(model, dataloaders=test_loader)
    
    expert_trainer.test(model, dataloaders=expert_loader)
    expert_trainer.predict(model, dataloaders=expert_loader)

if __name__ == '__main__':
    main()