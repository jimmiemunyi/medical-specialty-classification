import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from composer.models import ComposerModel

from torchmetrics import F1Score
from composer.metrics import CrossEntropy

from torchmetrics import Metric, MetricCollection
from typing import Any, Optional


class MedicalClassification(ComposerModel):
    def __init__(
        self,
        checkpoint,
        tokenizer,
        config_path=None,
        pretrained=True,
        num_labels=1,
        fc_dropout=0.2,
    ):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                checkpoint, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            # add tokenizer resizing here
            self.ptrn_model = AutoModel.from_pretrained(checkpoint, config=self.config)
            self.ptrn_model.resize_token_embeddings(len(tokenizer))
            self.model_forward_args = inspect.getfullargspec(
                self.ptrn_model.forward
            ).args
        else:
            self.ptrn_model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )
        self.train_metrics = MetricCollection(
            [CrossEntropy(), F1Score(task="multiclass", num_classes=num_labels)]
        )
        self.val_metrics = MetricCollection(
            [CrossEntropy(), F1Score(task="multiclass", num_classes=num_labels)]
        )
        self._init_weights(self.fc)
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.ptrn_model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k in self.model_forward_args}
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

    def loss(self, output, batch):
        labels = {k: v for k, v in batch.items() if k == "labels"}
        return F.cross_entropy(output, labels["labels"])

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        if outputs is not None:
            return outputs
        output = self.forward(batch)
        return output

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        labels = {k: v for k, v in batch.items() if k == "labels"}
        metric.update(outputs.squeeze(1), labels["labels"])
