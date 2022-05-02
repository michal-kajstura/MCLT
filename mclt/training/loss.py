import abc
from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from mclt.data.base import TaskDefinition


class BaseMultiTaskLoss(nn.Module, abc.ABC):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
    ):
        super().__init__()
        self._tasks = tasks
        self._loss_funcs = {
            name: nn.BCEWithLogitsLoss() if task.multilabel else nn.CrossEntropyLoss()
            for name, task in tasks.items()
        }

    @abc.abstractmethod
    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        pass


class MultiTaskLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
        weights: Optional[dict[str, float]] = None,
    ):
        super().__init__(tasks)
        self._weights = weights or {task: 1.0 for task in tasks}

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        loss = 0.0

        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            weight = self._weights[task_id]
            loss += self._loss_funcs[task_id](logits, labels) * weight

        return loss


class UncertaintyWeightedLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
    ):
        super().__init__(tasks)
        self._log_squared_variances = nn.ParameterDict(
            {task: nn.Parameter(data=torch.tensor([1.0], requires_grad=True)) for task in tasks}
        )

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        loss = 0.0

        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            log_sq_variance = self._log_squared_variances[task_id]
            inverted_variance = torch.exp(-log_sq_variance)
            loss += self._loss_funcs[task_id](logits, labels) * inverted_variance
            loss += log_sq_variance * 0.5
        return loss


class RandomWeightedLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
        distribution: Optional[torch.distributions.Distribution] = None,
    ):
        super().__init__(tasks)
        self._distribution = distribution or Normal(0.0, 1.0)

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        weights = self._distribution.sample((len(self._tasks),))
        weights = torch.softmax(weights, dim=-1)
        task_weights = {task: w for task, w in zip(self._tasks, weights)}

        loss = 0.0
        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            weight = task_weights[task_id]
            loss += self._loss_funcs[task_id](logits, labels) * weight

        return loss
