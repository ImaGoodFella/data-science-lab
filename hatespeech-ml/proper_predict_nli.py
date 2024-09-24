import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data.nli_dataset import get_dataloader

from pytorch_lightning.callbacks import ModelCheckpoint

from train.deepspeed_sequence import TextClassificationTransformer
from data.multi_gpu_pred_writer import CustomWriter

from peft import LoraConfig
from sophia import SophiaG

num_classes = 2
data_path = "../data/"

def get_data_loaders(model_name, model_nickname, batch_size, seed):

    test_loader = get_dataloader(
        model_name=model_name, 
        data_path=data_path, 
        dataset_name=f'test_{seed}.csv',
        batch_size=batch_size, 
        use_cache=True, 
        model_nickname=model_nickname, 
        comment_column='kommentar_original',
        label_column='hatespeech',
        shuffle=False,
        is_test=True
    )

    expert_loader = get_dataloader(
        model_name=model_name, 
        data_path=data_path, 
        dataset_name='expert_label_target.csv',
        batch_size=batch_size, 
        use_cache=True, 
        model_nickname=model_nickname, 
        comment_column='kommentar_original',
        label_column='hatespeech',
        shuffle=False,
        is_test=True
    )

    return test_loader, expert_loader

def get_trainers(model_nickname, optimizer_name, class_weights, batch_size, seed, max_epochs) -> (pl.Trainer, pl.Trainer):

    sub_folder_name = f'{model_nickname}/{optimizer_name}/cw_{int(class_weights)}/bs_{batch_size}/seed_{seed}'

    logger = pl_loggers.TensorBoardLogger(
        save_dir=data_path, name=f'nli_logs/{sub_folder_name}', 
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="train/F1Score",
        mode="max",
        dirpath=data_path + f"nli_checkpoint/{sub_folder_name}",
        filename="model-{epoch:02d}-{train/F1Score:.2f}",
        save_weights_only=False,
    )

    test_writer = CustomWriter(
        write_interval='epoch',
        output_file=data_path + f'nli_outputs/{sub_folder_name}/test_outputs.pkl'
    )

    expert_writer = CustomWriter(
        write_interval='epoch',
        output_file=data_path + f'nli_outputs/{sub_folder_name}/expert_outputs.pkl'
    )

    expert_trainer = pl.Trainer(
        accelerator="gpu", 
        strategy="deepspeed",
        devices='auto', 
        precision='16-mixed', 
        logger=False, 
        callbacks=[expert_writer], 
        log_every_n_steps=50
    )

    callbacks = [checkpoint_callback, test_writer]

    trainer = pl.Trainer(
        accelerator="gpu", 
        num_nodes=1,
        strategy='deepspeed', 
        devices='auto', 
        precision='16-mixed',
        logger=False, 
        max_epochs=max_epochs, 
        callbacks=callbacks, 
        log_every_n_steps=50
    )

    return trainer, expert_trainer

import argparse
from pathlib import Path

def main():

    class_weights = torch.tensor([1.0, 4.614545638159998]) #if args.class_weights else torch.tensor([1.0, 1.0])
    learning_rate = 5e-6
    batch_size = 16
    max_epochs = 3
    optimizer = None
    seed = 43
    lora = True

    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1) if lora else None
    
    model_nickname = 'e5_lora_nli_all_correct'
    model_name = "mjwong/multilingual-e5-large-xnli"

    optimizer = opti_params = None
    optimizer_name = 'adamw'

    trainer, expert_trainer = get_trainers(
        model_nickname=model_nickname, 
        optimizer_name=optimizer_name, 
        batch_size=batch_size, 
        seed=seed,
        max_epochs=max_epochs,
        class_weights=True,
    )

    test_loader, expert_loader = get_data_loaders(
        model_name=model_name,
        batch_size=batch_size,
        model_nickname=model_nickname,
        seed=seed,
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = TextClassificationTransformer(
        model_name, 
        num_labels=num_classes, 
        ignore_mismatched_sizes=True, 
        cache_dir="../data/huggingface", 
        deepspeed_sharding=True, 
        loss=criterion, 
        learning_rate=learning_rate, 
        weight_decay=0.01, 
        optimizer=optimizer,
        opti_params=opti_params,
        peft_config=peft_config
    )

    pl.seed_everything(seed, workers=True)
    
    #trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    ckpt_path = Path("/home/rasteiger/datasets/dslab/checkpoint/e5_lora_nli_all_correct/adamw/cw_1/bs_16/seed_43/model-epoch=02-train/F1Score=0.88.ckpt")
    #ckpt_path = Path("/home/rasteiger/dslab/data/nli_checkpoint/deberta/adamw/cw_1/bs_16/seed_43/model-epoch=02-train/F1Score=0.96.ckpt")
    #trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)
    trainer.predict(model, dataloaders=test_loader, ckpt_path=ckpt_path)
    
    #expert_trainer.test(model, dataloaders=expert_loader, ckpt_path=ckpt_path)
    expert_trainer.predict(model, dataloaders=expert_loader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()