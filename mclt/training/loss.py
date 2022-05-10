import abc
import random
from typing import Optional

import numpy as np
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


class UniformWeightedLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
        weights: Optional[dict[str, float]] = None,
    ):
        super().__init__(tasks)
        self._weights = weights or {task: 1.0 / len(tasks) for task in tasks}

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
            {
                task: nn.Parameter(data=torch.tensor([-0.5], requires_grad=True))
                for task in tasks
            }
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
            loss += (
                self._loss_funcs[task_id](logits, labels) * inverted_variance
                + log_sq_variance * 0.5
            )
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


class GradVaccineLoss(BaseMultiTaskLoss):
    def __init__(
        self,
        tasks: dict[str, TaskDefinition],
        beta: float = 1e-2,
    ):
        super().__init__(tasks)
        self._beta = beta
        self.task_num = len(tasks)
        self.rho_T = torch.zeros(self.task_num, self.task_num)

    def forward(
        self,
        task_groups: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        loss = 0.0
        for task_id, group in task_groups.items():
            logits = group['logits']
            labels = group['labels']
            loss += self._loss_funcs[task_id](logits, labels)
        return loss

    def backward(self, losses, **kwargs):
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')  # [task_num, grad_dim]
        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (
                        pc_grads[tn_i].norm() * grads[tn_j].norm())
                if rho_ij < self.rho_T[tn_i, tn_j]:
                    w = pc_grads[tn_i].norm() * (
                            self.rho_T[tn_i, tn_j] * (1 - rho_ij ** 2).sqrt() - rho_ij * (
                                1 - self.rho_T[tn_i, tn_j] ** 2).sqrt()) / (
                                grads[tn_j].norm() * (1 - self.rho_T[tn_i, tn_j] ** 2).sqrt())
                    pc_grads[tn_i] += grads[tn_j] * w
                    batch_weight[tn_j] += w.item()
                    self.rho_T[tn_i, tn_j] = (1 - self._beta) * self.rho_T[tn_i, tn_j] + self._beta * rho_ij
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count + 1)])
                param.grad.data = new_grads[beg:end].contiguous().view(
                    param.data.size()).data.clone()
            count += 1
