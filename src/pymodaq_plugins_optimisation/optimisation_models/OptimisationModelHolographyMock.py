from typing import List, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq_plugins_optimisation.utils import OptimisationModelGeneric, DataToActuatorOpti
from pymodaq_plugins_optimisation.hardware.gershberg_saxton import GBSAX

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataToExport, DataActuator, DataWithAxes, DataRaw
from pymodaq.utils.plotting.data_viewers import Viewer2D, ViewersEnum

from pymodaq.utils import gui_utils as gutils

from skimage.io import imread
from skimage.color import rgb2gray

if TYPE_CHECKING:
    from pymodaq_plugins_optimisation.extensions.optimisation import Optimisation

logger = set_logger(get_module_name(__file__))


class OptimisationModelHolographyMock(OptimisationModelGeneric):

    optimisation_algorithm = GBSAX()

    actuators_name = ["Shaper"]
    detectors_name = ["Camera Carac"]
    observables_dim = [ViewersEnum('Data2D'), ViewersEnum('Data2D')]

    params = [
        {'title': 'Target From:', 'name': 'target_source', 'type': 'list', 'limits': ['file']},
        {'title': 'File path:', 'name': 'target_file', 'type': 'browsepath', 'value': '', 'filetype': True},
        {'title': 'Send target to algo', 'name': 'send_target', 'type': 'bool_push', 'label': 'Send'},
        {'title': 'Apply Mask', 'name': 'apply_mask', 'type': 'bool',},
        {'title': 'Flip ud', 'name': 'flipud', 'type': 'bool', 'value': 'False'},
        {'title': 'Flip lr', 'name': 'fliplr', 'type': 'bool', 'value': 'True'},
        {'title': 'Move X', 'name': 'move_x', 'type': 'float', 'value': 0.},
        {'title': 'Move Y', 'name': 'move_y', 'type': 'float', 'value': 0.},
    ]

    def __init__(self, optimisation_controller: 'Optimisation'):
        super().__init__(optimisation_controller)

        self.other_detectors: List[str] = []
        self._temp_target_data: np.ndarray = None

        target_dock = gutils.Dock('Target')
        widget_target = QtWidgets.QWidget()
        target_dock.addWidget(widget_target)
        self.optimisation_controller.dockarea.addDock(target_dock, 'bottom',
                                                      self.optimisation_controller.docks['settings'])
        self.viewer_target = Viewer2D(widget_target)
        self.viewer_target.ROI_select_signal.connect(self.set_mask)
        self.mask: QtCore.QRectF = None

    def set_mask(self, mask: QtCore.QRectF):
        self.mask = mask

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
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
            else:
                self.get_target_from_file()
        elif param.name() == 'target_file':
            self.get_target_from_file()
        elif param.name() == 'send_target':
            self.load_target()

    def transform_image(self, data_in: DataWithAxes):
        data_out = data_in.deepcopy()
        if self.settings['flipud']:
            data_out = data_out.flipud()
        if self.settings['fliplr']:
            data_out = data_out.fliplr()
        return data_out

    def get_target_from_detector(self, dte: DataToExport):
        self._temp_target_data = dte.get_data_from_dim('Data2D')[0]
        self.viewer_target.show_data(self._temp_target_data)

    def get_target_from_file(self):
        fname = Path(self.settings['target_file'])
        if fname.is_file():
            try:
                img = np.flipud(imread(fname))
                if len(img.shape) == 2:
                    target_intensity = img
                elif len(img.shape) == 3:
                    target_intensity = rgb2gray(img[..., 0:3])
                self._temp_target_data = DataRaw('target', data=[target_intensity])
                self.viewer_target.show_data(self._temp_target_data)
            except Exception as e:
                logger.exception(str(e))

    def load_target(self):

        if self.settings['apply_mask']:
            data = self._temp_target_data.deepcopy()
            data.data[0] = np.zeros(data.data[0].shape)
            data.data[0][int(self.mask.y()): int(self.mask.y()+self.mask.height()),
                         int(self.mask.x()): int(self.mask.x()+self.mask.width())] = \
                self._temp_target_data.data[0][int(self.mask.y()): int(self.mask.y()+self.mask.height()),
                int(self.mask.x()): int(self.mask.x()+self.mask.width())]
            data = self.transform_image(data)
            self.optimisation_algorithm.load_target_data(data)
        else:
            data = self.transform_image(self._temp_target_data)
            self.optimisation_algorithm.load_target_data(data)

    def set_source(self):
        self.other_detectors = self.modules_manager.detectors_name
        for det in self.detectors_name:
            self.other_detectors.remove(det)

        source_limits: list = self.settings.child('target_source').opts['limits'].copy()
        source_limits.extend(self.other_detectors)

        self.settings.child('target_source').setLimits(source_limits)

    def ini_model(self):
        super().ini_models()
        self.set_source()

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
        phase = outputs[0]
        phase = self.add_linear_phase(phase, self.settings['move_y'], self.settings['move_x'])
        return DataToActuatorOpti('outputs', mode='abs', data=[DataActuator(self.actuators_name[0], data=phase)])

    def add_linear_phase(self, phase: np.ndarray, move_y: float, move_x: float):
        linphase_x = np.linspace(0, 2*np.pi, phase.shape[1]) * move_x * 10
        linphase_y = np.linspace(0, 2*np.pi, phase.shape[0]) * move_y * 10

        linear_phase = np.zeros(phase.shape)
        for ind in range(phase.shape[0]):
            linear_phase[ind, :] = linphase_x + linphase_y[ind]
        return self.wrap(linear_phase + phase)

    def wrap(self, x):
        return (x-np.min(x)) % (2*np.pi)


if __name__ == '__main__':
    pass


