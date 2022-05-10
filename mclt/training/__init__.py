from mclt.training.loss import (
    UniformWeightedLoss, RandomWeightedLoss, UncertaintyWeightedLoss,
    GradVaccineLoss,
)

LOSS_FUNCS = {
    'uniform': UniformWeightedLoss,
    'random': RandomWeightedLoss,
    'uncertainty': UncertaintyWeightedLoss,
    'grad_vaccine': GradVaccineLoss,
}
