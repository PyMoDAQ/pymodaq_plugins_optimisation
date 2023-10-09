from typing import List, Union
from pathlib import Path

import numpy as np


from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataToExport, DataActuator, DataWithAxes, DataRaw

from pymodaq_plugins_optimisation.optimisation_models.OptimisationModelHolographyMock import \
    OptimisationModelHolographyMock, DataToActuatorOpti, OptimisationModelGeneric

logger = set_logger(get_module_name(__file__))


class OptimisationModelHolography(OptimisationModelHolographyMock):

    def __init__(self, optimisation_controller):
        super().__init__(optimisation_controller)
        self.phase_polyfit: np.polynomial.Polynomial = None

    def ini_model(self):
        OptimisationModelGeneric.ini_models(self)
        self.set_source()
        self.optimisation_algorithm.load_image()

        calib = np.load(Path(r'C:\Data\2023\calibration_holography.npy'))
        grey = calib[0, :]
        phase = calib[1, :]
        self.phase_polyfit = np.polynomial.Polynomial.fit(phase, grey, 11)

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

    def convert_output(self, outputs: List[np.ndarray]) -> DataToActuatorOpti:
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
        phase = outputs[0]
        phase_wraped = self.wrap(phase)
        phase_linear = self.add_linear_phase(phase_wraped, self.settings['move_y'], self.settings['move_x'])
        grey_levels = self.phase_polyfit(phase_linear)
        induced_amplitude = None
        self.optimisation_algorithm.set_phase_in_object_plane(phase_wraped, induced_amplitude=induced_amplitude)
        return DataToActuatorOpti('outputs', mode='abs', data=[
            DataActuator(self.actuators_name[0], data=grey_levels)])


if __name__ == '__main__':
    pass


