import abc
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch import Tensor, nn
from torch.optim import AdamW
from torchmetrics import F1Score, MetricCollection
from transformers import get_linear_schedule_with_warmup

from mclt.data.base import TaskDefinition
from mclt.modeling.multi_classifier import MultiTaskTransformer
from mclt.training import UncertaintyWeightedLoss


class MultitaskTransformerTrainer(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        model: MultiTaskTransformer,
        tasks: dict[str, TaskDefinition],
        learning_rate: float = 1e-5,
        warmup_steps_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self._learning_rate = learning_rate
        self._warmup_steps_ratio = warmup_steps_ratio
        self._weight_decay = weight_decay
        self.model = model
        self.tasks = tasks

        metrics = {
            task_name: MetricCollection(
                {
                    f'{task_name}/f1_score': F1Score(num_classes=task.num_labels),
                    f'{task_name}/f1_score_macro': F1Score(
                        num_classes=task.num_labels, average='macro'
                    ),
                }
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

    def training_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> dict[str, Tensor]:
        if (scheduler := self.lr_schedulers()) is not None:
            scheduler.step()  # type: ignore
        return self._step(batch, 'train_set', 'train')['loss']

    def validation_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        *args,
    ):
        dataset_idx = args[0] if args else 0
        o = self._step(batch, 'val_set', self._task_names[dataset_idx])
        return o['loss']

    def test_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        *args,
    ):
        dataset_idx = args[0] if args else 0
        return self._step(batch, 'test_set', self._task_names[dataset_idx])['loss']

    def _epoch_end(
        self,
        outs,
        step_type,
    ):
        f1_scores = []
        for metrics in self._metrics[step_type].values():
            to_log = metrics.compute()
            self.log_dict(to_log)
            metrics.reset()
            f1_scores.extend(v for k, v in to_log.items() if k.endswith('f1_score'))

        global_f1 = torch.stack(f1_scores).mean()
        self.log(f'{step_type}/f1_score', global_f1)

    def training_epoch_end(self, outputs) -> None:
        self._epoch_end(outputs, 'train_set')

    def validation_epoch_end(self, outputs) -> None:
        self._epoch_end(outputs, 'val_set')

    def test_epoch_end(self, outputs) -> None:
        self._epoch_end(outputs, 'test_set')

    def configure_optimizers(self):
        optimizer = AdamW(
            params=
                self.model.parameters(),
            # *[
            #     {'params': classifier.parameters(), 'name': name}
            #     for name, classifier
            #     in self.model._multi_task_head._classifiers.items()
            # ],
            # {'params': self.model._transformer.parameters(), 'name': 'backbone'},
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
        dataset_name: str,
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

        loss_func = self.model.loss_func
        if isinstance(loss_func, UncertaintyWeightedLoss):
            self.log_dict(
                {f'{step_type}/{dataset_name}/{k}_weight': v.data.item() for k, v in loss_func._log_squared_variances.items()},
                add_dataloader_idx=False,
            )

        self.log('train/loss', loss)
        return {'loss': loss}


class GradSurgeryTrainer(MultitaskTransformerTrainer):

    def __init__(
        self,
        model: MultiTaskTransformer,
        tasks: dict[str, TaskDefinition],
        learning_rate: float = 1e-5,
        warmup_steps_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ):
        super().__init__(
            model=model,
            tasks=tasks,
            learning_rate=learning_rate,
            warmup_steps_ratio=warmup_steps_ratio,
            weight_decay=weight_decay,
        )
        self.automatic_optimization = False

    def training_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> dict[str, Tensor]:
        losses = super().training_step(batch, batch_idx)
        optimizer = self.optimizers().optimizer
        optimizer.zero_grad()

        # l = []
        scaler = self.trainer.accelerator.precision_plugin.scaler
        scaled_losses = [
            scaler.scale(loss) for loss in losses.values()
        ]

        # scaler.unscale_(optimizer)
        self.model.loss_func.backward(scaled_losses, optimizer, scaler)

        self.log_dict(losses)

        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()


class AdapterTuneTrainer(MultitaskTransformerTrainer):

    def __init__(
        self,
        model: MultiTaskTransformer,
        tasks: dict[str, TaskDefinition],
        learning_rate: float = 1e-5,
        warmup_steps_ratio: float = 0.1,
        weight_decay: float = 0.01,
        adapter_learning_rate: float = 1e-4,
        adapter_finetune_steps_ratio: float = 0.1,
    ):
        super().__init__(
            model=model,
            tasks=tasks,
            learning_rate=learning_rate,
            warmup_steps_ratio=warmup_steps_ratio,
            weight_decay=weight_decay,
        )
        self._adapter_learning_rate = adapter_learning_rate
        self._shared_training_steps_ratio = 1.0 - adapter_finetune_steps_ratio
        self._adapter_finetune_steps_ratio = adapter_finetune_steps_ratio / len(tasks)
        self._adapter_tuning_phase = False

    def training_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> dict[str, Tensor]:
        if not self._adapter_tuning_phase and self.global_step >= self._shared_training_steps_ratio:
            t: Trainer = self.trainer
            t.reset_train_dataloader()

        return super().training_step(batch, batch_idx)

    def on_train_epoch_start(self) -> None:
        pass