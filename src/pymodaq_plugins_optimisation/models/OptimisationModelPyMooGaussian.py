from typing import List, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq_plugins_optimisation.utils import OptimisationModelGeneric, DataToActuatorOpti
from pymodaq_plugins_optimisation.hardware.gershberg_saxton import GBSAX

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataToExport, DataActuator, DataRaw
from pymodaq.utils.plotting.data_viewers.viewer import ViewersEnum

from pymodaq.utils import gui_utils as gutils
from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.termination import NoTermination


if TYPE_CHECKING:
    from pymodaq_plugins_optimisation.extensions.optimisation import Optimisation

logger = set_logger(get_module_name(__file__))


class GaussianProblem(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=1, n_ieq_constr=0, xl=0.0, xu=256.0)

    def _evaluate(self, x, out, *args, **kwargs):
        ...


class OptimisationModelPyMooGaussian(OptimisationModelGeneric):

    optimisation_algorithm: Algorithm = NelderMead()

    actuators_name = ["Xaxis"]
    detectors_name = ["Gaussian"]
    observables_dim = [ViewersEnum('Data0D')]

    params = []

    def __init__(self, optimisation_controller: 'Optimisation'):
        super().__init__(optimisation_controller)

        self.problem = None
        self.other_detectors: List[str] = []

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        """
        pass

    def ini_model(self):
        super().ini_models()
        self.problem = GaussianProblem()
        self.optimisation_algorithm = NSGA2(pop_size=10)
        self.optimisation_algorithm.setup(self.problem, termination=NoTermination(), seed=1, verbose=True)

    def convert_input(self, measurements: DataToExport) -> DataToExport:
        """
        Convert the measurements in the units to be fed to the Optimisation Controller
        Parameters
        ----------
        measurements: DataToExport
            data object exported from the detectors from which the model extract a value of the same units as
            the setpoint

        Returns
        -------
        DataToExport

        """
        fitness = measurements.data[0].data[0]
        return DataToExport('evaluation', data=[DataRaw('data', data=[-fitness])])

    def convert_output(self, outputs: List[np.ndarray]) -> DataToActuatorOpti:
        """
        Convert the output of the Optimisation Controller in units to be fed into the actuators
        Parameters
        ----------
        outputs: list of numpy ndarray
            output value from the controller from which the model extract a value of the same units as the actuators

        Returns
        -------
        DataToActuatorOpti: derived from DataToExport. Contains value to be fed to the actuators with a mode
            attribute, either 'rel' for relative or 'abs' for absolute.

        """
        data = outputs[0]
        return DataToActuatorOpti('outputs', mode='abs', data=[DataActuator(self.actuators_name[0], data=data)])


if __name__ == '__main__':
    pass


