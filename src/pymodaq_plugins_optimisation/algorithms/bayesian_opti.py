from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

import numpy as np

from pymodaq_plugins_optimisation.utils import OptimisationAlgorithm


class Algorithm(OptimisationAlgorithm):

    def __init__(self, ini_random: int, bounds: dict, **kwargs):

        self._algo = BayesianOptimization(f=None,
                                          random_state=ini_random,
                                          pbounds=bounds,
                                          **kwargs
                                          )

        self._utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    def ask(self) -> np.ndarray:
        self._next_points = self._algo.space.params_to_array(self._algo.suggest(self._utility))
        return self._next_points

    def tell(self, function_value: float):
        self._algo.register(params=self._next_points, target=function_value)

    @property
    def best_fitness(self) -> float:
        return self._algo.max['target']

    @property
    def best_individual(self) -> np.ndarray:
        return self._algo.space.params_to_array(self._algo.max['params'])
