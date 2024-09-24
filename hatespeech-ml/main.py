import torch
import itertools

from transformers import AutoModelForSequenceClassification

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data.dataset import get_dataloaders
from train.sequence import ClassificationWrapper

num_classes = 2

model_name = "unitary/multilingual-toxic-xlm-roberta"

data_path = "../data/"

class_weights = [torch.tensor([1.0, 4.60967657991111]), None]
learning_rates = [1e-5, 5e-5, 1e-4]
batch_sizes = [8, 16, 32]

callbacks = []

def main():

    for batch_size, learning_rate, class_weight in itertools.product(batch_sizes, learning_rates, class_weights):

        version_name = f'version_bs={batch_size}_lr={learning_rate}_cw={class_weight}'

        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        logger = logger=pl_loggers.TensorBoardLogger(save_dir=data_path, name='logs/initial', version=version_name)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
        wrapper = ClassificationWrapper(model=model, learning_rate=learning_rate, weight_decay=0.00, loss=criterion, num_classes=num_classes)

        trainer = pl.Trainer(accelerator="gpu", devices=4, logger=logger, max_epochs=3, callbacks=callbacks, precision='16-mixed')

        train_loader, eval_loader = get_dataloaders(model_name=model_name, data_path=data_path, batch_size=batch_size)
        trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=eval_loader)
        
if __name__ == '__main__':
    main()