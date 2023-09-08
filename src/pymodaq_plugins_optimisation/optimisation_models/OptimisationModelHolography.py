from typing import List, Union
from pathlib import Path

import numpy as np

from pymodaq_plugins_optimisation.utils import OptimisationModelGeneric
from pymodaq_plugins_optimisation.hardware.gershberg_saxton import GBSAX
from pymodaq.utils.managers.modules_manager import ModulesManager
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataToExport, DataActuator, DataWithAxes
from pymodaq.extensions.pid.utils import DataToActuatorPID
from skimage.io import imread
from skimage.color import rgb2gray

logger = set_logger(get_module_name(__file__))


class OptimisationModelHolography(OptimisationModelGeneric):

    optimisation_algorithm = GBSAX()

    actuators_name = ["Shaper"]
    detectors_name = ['Camera']

    params = []

    def __init__(self, optimisation_controller):
        self.opti_controller = optimisation_controller  # instance of the pid_controller using this model
        self.modules_manager: ModulesManager = optimisation_controller.modules_manager

        self.settings = self.opti_controller.settings.child('models', 'model_params')  # set of parameters
        self.check_modules(self.modules_manager)

    def check_modules(self, modules_manager):
        for act in self.actuators_name:
            if act not in modules_manager.actuators_name:
                logger.warning(f'The actuator {act} defined in the PID model is not present in the Dashboard')
                return False
        for det in self.detectors_name:
            if det not in modules_manager.detectors_name:
                logger.warning(f'The detector {det} defined in the PID model is not present in the Dashboard')

    def update_detector_names(self):
        names = self.opti_controller.settings.child('main_settings', 'detector_modules').value()['selected']
        self.data_names = []
        for name in names:
            name = name.split('//')
            self.data_names.append(name)

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        To be overwritten in child class
        """
        if param.name() == '':
            pass

    def ini_model(self):
        self.modules_manager.selected_actuators_name = self.actuators_name
        self.modules_manager.selected_detectors_name = self.detectors_name
        self.optimisation_algorithm: GBSAX = self.modules_manager.actuators[0].controller
        self.optimisation_algorithm.load_image()

    def convert_input(self, measurements: DataToExport):
        """
        Convert the measurements in the units to be fed to the Optimisation Controller
        Parameters
        ----------
        measurements: DataToExport
            data object exported from the detectors from which the model extract a value of the same units as
            the setpoint

        Returns
        -------
        float: the converted input

        """
        return DataToExport('inputs', data=[measurements.get_data_from_name('GBSAX Intensity')])

    def convert_output(self, outputs: List[np.ndarray]) -> DataToActuatorPID:
        """
        Convert the output of the Optimisation Controller in units to be fed into the actuators
        Parameters
        ----------
        outputs: list of numpy ndarray
            output value from the controller from which the model extract a value of the same units as the actuators

        Returns
        -------
        DataToActuatorPID: derived from DataToExport. Contains value to be fed to the actuators with a a mode
            attribute, either 'rel' for relative or 'abs' for absolute.

        """
        return DataToActuatorPID('outputs', mode='abs', data=[DataActuator(self.actuators_name[ind], data=outputs[ind])
                                                              for ind in range(len(outputs))])

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        Parameters
        ----------
        param: (Parameter) instance of Parameter object
        """
        if param.name() == '':
            pass


if __name__ == '__main__':
    pass


