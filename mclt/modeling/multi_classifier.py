from typing import Any, Optional

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from mclt.data.base import TaskDefinition
from mclt.training.loss import BaseMultiTaskLoss, UniformWeightedLoss
from mclt.utils.grouping import group_by_task


class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        transformer: PreTrainedModel,
        tasks: dict[str, TaskDefinition],
        loss_func: Optional[BaseMultiTaskLoss] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self._multi_task_head = MultiTaskHead(
            model_dim=transformer.config.hidden_size,
            task_num_labels={k: t.num_labels for k, t in tasks.items()},
        )
        self.loss_func = loss_func or UniformWeightedLoss(tasks)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        task_ids: list[str],
        labels: Optional[Tensor] = None,
    ) -> dict[str, Any]:
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = transformer_output.last_hidden_state

        outputs_per_task = self._multi_task_head(
            last_hidden_state=last_hidden_state,
            task_ids=task_ids,
        )

        outputs_per_task = {k: {'logits': v['logits']} for k, v in outputs_per_task.items()}

        if labels is not None:
            labels_per_task = group_by_task(task_ids, labels=labels)
            for task in labels_per_task:
                outputs_per_task[task]['labels'] = labels_per_task[task]['labels']
            outputs_per_task['loss'] = self.loss_func(outputs_per_task)

        return outputs_per_task


class MultiTaskHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        task_num_labels: dict[str, int],
    ):
        super().__init__()

        self._classifiers = nn.ModuleDict(
            {
                task: ClassificationHead(model_dim, num_labels)
                for task, num_labels in task_num_labels.items()
            }
        )

    def forward(
        self,
        last_hidden_state: Tensor,
        task_ids: list[str],
    ) -> dict[str, Any]:
        input_to_task = group_by_task(task_ids, last_hidden_state=last_hidden_state)

        for task_id, batch in input_to_task.items():
            batch['logits'] = self._classifiers[task_id](batch['last_hidden_state'])

        return input_to_task


# copied from modeling_roberta.py
class ClassificationHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_labels: int,
        classifier_dropout: float = 0.1,
    ):
        super().__init__()
        self._dense = nn.Linear(model_dim, model_dim)
        self._dropout = nn.Dropout(classifier_dropout)
        self._out_proj = nn.Linear(model_dim, num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self._dropout(x)
        x = self._dense(x)
        x = torch.tanh(x)
        x = self._dropout(x)
        x = self._out_proj(x)
        return x
