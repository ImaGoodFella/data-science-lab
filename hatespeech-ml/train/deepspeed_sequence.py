from typing import Any, Dict, Type, Callable, Optional
import os

import torch
import torch.nn as nn

import transformers
from torchmetrics.classification import F1Score, Precision, Recall, Accuracy
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoConfig

from lightning_transformers.core import TaskTransformer
from lightning_transformers.utilities.deepspeed import enable_transformers_pretrained_deepspeed_sharding

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType 

class TextClassificationTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Text Classification Task.

    Args:
        *args: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSequenceClassification``)
        **kwargs: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        loss: Callable,
        learning_rate: float = 1e-4,
        weight_decay : float = 1e-6,
        optimizer = None,
        opti_params = None,
        peft_config = None,
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForSequenceClassification,
        **kwargs,

    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        self.num_classes = kwargs['num_labels']
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = loss
        self.optimizer = optimizer
        self.opti_params = opti_params
        self.peft_config = peft_config
        self.set_metrics()
        
        if hasattr(self, "model") and self.peft_config:
            self.model = get_peft_model(self.model, self.peft_config)
            self.model.print_trainable_parameters()

    def setup(self, stage: Optional[str] = None) -> None:
        self.configure_metrics(stage)
        if self.deepspeed_sharding and not hasattr(self, "model"):
            enable_transformers_pretrained_deepspeed_sharding(self)
            self.initialize_model(self.pretrained_model_name_or_path)
            
            if self.peft_config:
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()

    def initialize_model(self, pretrained_model_name_or_path: str):
        """create and initialize the model to use with this task,

        Feel free to overwrite this method if you are initializing the model in a different way
        """
        if self.load_weights:
            self.model = self.downstream_model_type.from_pretrained(
                pretrained_model_name_or_path, **self.model_data_kwargs
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model_data_kwargs["pretrained_model"], **self.model_data_kwargs
            )
            self.model = self.downstream_model_type.from_config(config)
        
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
        return self.forward(**batch), batch['labels']
    
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
        
        if self.optimizer is not None: 
            optimizer = self.optimizer(self.parameters(), **self.opti_params)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=400,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }