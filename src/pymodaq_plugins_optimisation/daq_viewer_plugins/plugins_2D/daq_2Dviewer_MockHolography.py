from pathlib import Path

import numpy as np

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter

from pymodaq_plugins_optimisation.hardware.gershberg_saxton import GBSAX


class DAQ_2DViewer_MockHolography(DAQ_Viewer_base):
    """
    """
    params = comon_parameters + [
        {'title': 'Browse Image', 'name': 'browse_image', 'type': 'browsepath', 'filetype': True},
        {'title': 'Load Target Image', 'name': 'load_image', 'type': 'bool_push', 'label': 'Load Image'},
        {'title': 'Evolve me!', 'name': 'evolve', 'type': 'bool_push', 'label': 'Evolve me!'}

    ]

    def ini_attributes(self):
        self.controller: GBSAX = None

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        # TODO for your custom plugin
        if param.name() == "load_image":
            fname = Path(self.settings['browser_image'])
            if fname.is_file():
                self.controller.load_image(fname)
            else:
                self.controller.load_image()
        elif param.name() == 'evolve':
            self.controller.propagate_field()
            self.controller.evolve_field()

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        self.ini_detector_init(old_controller=controller,
                               new_controller=GBSAX())

        if self.controller.target_intensity is None:
            self.controller.load_image()

        self.dte_signal_temp.emit(DataToExport('GBSAX',
                                               data=[DataFromPlugins(name='GBSAX Intensity',
                                                                     data=[np.abs(self.controller.field_image)**2],
                                                                     dim='Data2D', labels=['Field Object intensity']),
                                                     ]))

        info = "GBSAX initialized"
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        pass

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """

        self.dte_signal.emit(DataToExport('GBSAX',
                                          data=[DataFromPlugins(name='GBSAX Intensity',
                                                                data=[np.abs(self.controller.field_image)**2],
                                                                dim='Data2D', labels=['Field Object intensity']),
                                                ]))

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        pass
        return ''


if __name__ == '__main__':
    main(__file__)
