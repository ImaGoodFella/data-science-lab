from typing import Callable, Tuple
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import F1Score, Precision, Recall, Accuracy

class ClassificationWrapper(pl.LightningModule):
    """
    A Pytorch Lightning Wrapper to train
    multimodal classification models
    """

    def __init__(self, 
                 model : nn.Module,
                 loss: Callable,
                 learning_rate: float = 1e-4,
                 weight_decay : float = 1e-6,
                 num_classes : int = 2,
                 optimizer = None,
                 opti_params = None,
    ) -> None:
        """
        Constructer for ClassificationWrapper
        model : nn.Module,
            The model to train
        loss: Callable = nn.CrossEntropyLoss()
            The loss function to use
        learning_rate: float = 1e-4,
            The learning rate for the optimizer
        weight_decay : float = 1e-6,
            The weight decay for the optimizer
        """
    
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = loss
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.opti_params = opti_params
        self.set_metrics()

    def set_metrics(self):
        
        # Set arguments for metrics depending if we have a binary or multi classification problem
        if self.num_classes == 2:
            args = {'task' : 'binary', 'average' : 'macro'}
        else:
            args = {'task' : 'multiclass', 'average' : 'macro'}
        
        self.train_metrics = nn.ModuleDict({
            "F1Score": F1Score(**args),
            "Precision": Precision(**args),
            "Recall": Recall(**args),
            "Accuracy": Accuracy(**args),
        })

        self.eval_metrics = nn.ModuleDict({
            "F1Score": F1Score(**args),
            "Precision": Precision(**args),
            "Recall": Recall(**args),
            "Accuracy": Accuracy(**args),
        })

        self.train_metrics_neg = nn.ModuleDict({
            "F1Score": F1Score(**args),
            "Precision": Precision(**args),
            "Recall": Recall(**args),
            "Accuracy": Accuracy(**args),
        })

        self.eval_metrics_neg = nn.ModuleDict({
            "F1Score": F1Score(**args),
            "Precision": Precision(**args),
            "Recall": Recall(**args),
            "Accuracy": Accuracy(**args),
        })

    def predict_step(self, batch, batch_idx, data_loader_idx=0):
        return self.forward(**batch)
    
    def forward(self, **inputs) -> torch.Tensor:
        return self.model(**inputs)
    
    def get_pred(self, logits):
        if hasattr(self.model, "predict"):
            return self.model.predict(logits)
        else:
            return torch.argmax(logits, dim=1)
    
    def standard_step(self, batch, batch_idx, stage_name, metric_dict, metric_dict_neg):

        labels = batch['labels']
        outputs = self.model(**batch)
        logits = outputs['logits']
        
        loss = self.loss(logits, labels)

        self.log(f"{stage_name}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        preds = self.get_pred(logits)
        for name, metric in metric_dict.items():
            metric.update(preds.detach(), labels.detach())
            self.log(f'{stage_name}/{name}', metric, on_epoch=True, on_step=False, logger=True, sync_dist=True)

        for name, metric in metric_dict_neg.items():
            metric.update(1 - preds.detach(), 1 - labels.detach())
            self.log(f'{stage_name}/neg_{name}', metric, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.standard_step(batch, batch_idx, 'train', self.train_metrics, self.train_metrics_neg)

    def validation_step(self, batch, batch_idx):
        return self.standard_step(batch, batch_idx, 'val', self.eval_metrics, self.eval_metrics_neg)
    
    def test_step(self, batch, batch_idx):
        return self.standard_step(batch, batch_idx, 'test', self.eval_metrics, self.eval_metrics_neg)
    
    def configure_optimizers(self):
        
        if self.optimizer is not None: return self.optimizer(self.parameters(), **self.opti_params)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer