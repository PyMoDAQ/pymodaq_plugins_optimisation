from typing import List, Union
from pathlib import Path

import numpy as np

from pymodaq_plugins_optimisation.utils import OptimisationModelGeneric
from pymodaq_plugins_optimisation.hardware.gershberg_saxton import GBSAX
from pymodaq.utils.managers.modules_manager import ModulesManager
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataToExport, DataActuator, DataWithAxes, DataRaw
from pymodaq.extensions.pid.utils import DataToActuatorPID
from skimage.io import imread
from skimage.color import rgb2gray

logger = set_logger(get_module_name(__file__))


class OptimisationModelHolographyMock(OptimisationModelGeneric):

    optimisation_algorithm = GBSAX()

    actuators_name = ["Shaper"]
    detectors_name = ["Camera Carac"]

    params = [
        {'title': 'Target From:', 'name': 'target_source', 'type': 'list', 'limits': ['file']},
        {'title': 'File path:', 'name': 'target_file', 'type': 'browsepath', 'value': '', 'filetype': True},
    ]

    def __init__(self, optimisation_controller):
        self.optimisation_controller = optimisation_controller  # instance of the pid_controller using this model
        self.modules_manager: ModulesManager = optimisation_controller.modules_manager

        self.settings = self.optimisation_controller.settings.child('models', 'model_params')  # set of parameters
        self.check_modules(self.modules_manager)
        self.other_detectors: List[str] = []
        self._temp_target_data: np.ndarray = None

    def check_modules(self, modules_manager):
        for act in self.actuators_name:
            if act not in modules_manager.actuators_name:
                logger.warning(f'The actuator {act} defined in the PID model is not present in the Dashboard')
                return False
        for det in self.detectors_name:
            if det not in modules_manager.detectors_name:
                logger.warning(f'The detector {det} defined in the PID model is not present in the Dashboard')

    def update_detector_names(self):
        names = self.optimisation_controller.settings.child('main_settings', 'detector_modules').value()['selected']
        self.data_names = []
        for name in names:
            name = name.split('//')
            self.data_names.append(name)

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        To be overwritten in child class
        """
        if param.name() == 'target_source':
            if param.value() != 'file' and param.value() in self.other_detectors:

                for det_name in self.other_detectors:
                    det = self.modules_manager.get_mod_from_name(det_name, 'det')
                    try:
                        det.grab_done_signal.disconnect(self.get_target_from_detector)
                    except:
                        pass
                det = self.modules_manager.get_mod_from_name(param.value(), 'det')
                det.grab_done_signal.connect(self.get_target_from_detector)
        elif param.name() == 'target_file':
            self.load_target()

    def get_target_from_detector(self, dte: DataToExport):
        self._temp_target_data = dte
        self.load_target()

    def load_target(self):
        if self.settings['target_source'] == 'file':
            self.optimisation_algorithm.load_image(self.settings['target_file'])
        else:
            self.optimisation_algorithm.load_target_data(self._temp_target_data)

    def ini_model(self):
        self.modules_manager.selected_actuators_name = self.actuators_name
        self.modules_manager.selected_detectors_name = self.detectors_name

        self.other_detectors = self.modules_manager.detectors_name
        for det in self.detectors_name:
            self.other_detectors.remove(det)

        source_limits: list = self.settings.child('target_source').opts['limits'].copy()
        source_limits.extend(self.other_detectors)

        self.settings.child('target_source').setLimits(source_limits)
        self.optimisation_algorithm: GBSAX = self.modules_manager.actuators[0].controller  # specific of the MockModel

        self.optimisation_algorithm.load_image()

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
        #  in here I'm not using the result of a measurement but the field calculated using FFT
        return DataToExport('inputs', data=[
            DataRaw('field image', data=[np.abs(self.optimisation_algorithm.field_image)**2])])

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



if __name__ == '__main__':
    pass


