import abc
from typing import Dict

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics import F1Score, MetricCollection
from transformers import AdamW, get_linear_schedule_with_warmup

from mclt.data.base import TaskDefinition
from mclt.modeling.multi_classifier import MultiTaskTransformer


class BaseSentimentTrainer(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        learning_rate: float = 1e-5,
        warmup_steps_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self._learning_rate = learning_rate
        self._warmup_steps_ratio = warmup_steps_ratio
        self._weight_decay = weight_decay

    def training_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> dict[str, Tensor]:
        self.lr_schedulers().step()  # type: ignore
        return self._step(batch, 'train_set', batch_idx, 0)['loss']

    def validation_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        *args,
    ):
        dataset_idx = args[0] if args else 0
        o = self._step(batch, 'val_set', batch_idx, dataset_idx)
        return o['loss']

    def test_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        *args,
    ):
        dataset_idx = args[0] if args else 0
        return self._step(batch, 'test_set', batch_idx, dataset_idx)['loss']

    def _epoch_end(
        self,
        outs,
        step_type,
    ):
        all_metrics = []
        for metrics in self._metrics[step_type].values():
            to_log = metrics.compute()
            self.log_dict(to_log)
            metrics.reset()
            all_metrics.extend(to_log.values())

        global_f1 = torch.stack(all_metrics).mean()
        self.log(f'{step_type}/f1_score', global_f1)

    def training_epoch_end(self, outputs) -> None:
        self._epoch_end(outputs, 'train_set')

    def validation_epoch_end(self, outputs) -> None:
        self._epoch_end(outputs, 'val_set')

    def test_epoch_end(self, outputs) -> None:
        self._epoch_end(outputs, 'test_set')

    @abc.abstractmethod
    def forward(  # type: ignore
        self,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def _step(
        self,
        batch: Dict[str, Tensor],
        step_type: str,
        batch_idx: int,
        dataset_idx: int,
    ):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        if self.trainer.max_steps:
            num_training_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        else:
            num_training_steps = (
                self.trainer.max_epochs
                * len(self.trainer._data_connector._train_dataloader_source.dataloader())
                * self.trainer.accumulate_grad_batches
            )

        num_warmup_steps = int(num_training_steps * self._warmup_steps_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }


class MultitaskTransformerTrainer(BaseSentimentTrainer):
    def __init__(
        self,
        model: MultiTaskTransformer,
        tasks: dict[str, TaskDefinition],
        learning_rate: float = 1e-5,
        warmup_steps_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ):
        super().__init__(
            learning_rate=learning_rate,
            warmup_steps_ratio=warmup_steps_ratio,
            weight_decay=weight_decay,
        )
        self.model = model
        metrics = {
            task_name: MetricCollection(
                {f'{task_name}/f1_score': F1Score(num_classes=task.num_labels)}
            )
            for task_name, task in tasks.items()
        }
        self._metrics = nn.ModuleDict(
            {
                set_name: nn.ModuleDict(
                    {
                        task_name: metrics.clone(f'{set_name}/')
                        for task_name, metrics in metrics.items()
                    }
                )
                for set_name in ('train_set', 'val_set', 'test_set')
            }
        )
        self._task_names = list(tasks)

    def forward(  # type: ignore
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        task_ids,
        labels=None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_ids=task_ids,
            labels=labels,
        )

    def _step(
        self,
        batch: Dict[str, Tensor],
        step_type: str,
        batch_idx: int,
        dataset_idx: int,
    ):
        task_ids = batch['task']
        labels = batch['labels']
        outputs_per_task = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            task_ids=task_ids,
            labels=labels,
        )

        loss = outputs_per_task.pop('loss')
        for task, group in outputs_per_task.items():
            logits = group['logits']
            labels = group['labels'].int()
            self._metrics[step_type][task].update(logits, labels)

        self.log(f'{step_type}/loss', loss, on_epoch=True)
        return {'loss': loss}
