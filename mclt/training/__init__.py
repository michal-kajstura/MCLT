from mclt.training.loss import (
    UniformWeightedLoss, RandomWeightedLoss, UncertaintyWeightedLoss,
    GradVaccineLoss, GradSurgeryLoss,
)

LOSS_FUNCS = {
    'uniform': UniformWeightedLoss,
    'random': RandomWeightedLoss,
    'uncertainty': UncertaintyWeightedLoss,
    'grad_vaccine': GradVaccineLoss,
    'grad_surgery': GradSurgeryLoss,
}
