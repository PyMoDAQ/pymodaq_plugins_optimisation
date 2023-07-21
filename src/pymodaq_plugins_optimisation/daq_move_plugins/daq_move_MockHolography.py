import numpy as np

from pymodaq.control_modules.move_utility_classes import DAQ_Move_base, comon_parameters_fun, main  # common set of parameters for all actuators
from pymodaq.utils.daq_utils import ThreadCommand  # object used to send info back to the main thread
from pymodaq.utils.parameter import Parameter

from pymodaq.utils.config import Config
config = Config()

from pymodaq_plugins_optimisation.hardware.gershberg_saxton import GBSAX


class DAQ_Move_MockHolography(DAQ_Move_base):
    """Plugin for the Template Instrument

    This object inherits all functionality to communicate with PyMoDAQ Module through inheritance via DAQ_Move_base
    It then implements the particular communication with the instrument

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library
    """
    _controller_units = 'rad'
    is_multiaxes = False
    axes_names = ['Axis1']
    _epsilon = 0.1
    params = [
                ] + comon_parameters_fun(is_multiaxes, axes_names, epsilon=_epsilon)
    # _epsilon is the initial default value for the epsilon parameter allowing pymodaq to know if the controller reached
    # the target value. It is the developer responsibility to put here a meaningful value

    def ini_attributes(self):
        self.controller: GBSAX = None

    def get_actuator_value(self):
        """Get the current value from the hardware with scaling conversion.

        Returns
        -------
        float: The position obtained after scaling conversion.
        """
        pos = np.angle(self.controller.field_object)  # when writing your own plugin replace this line
        return pos

    def close(self):
        """Terminate the communication protocol"""
        pass

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "a_parameter_you've_added_in_self.params":
           self.controller.your_method_to_apply_this_param_change()
        else:
            pass

    def ini_stage(self, controller=None):
        """Actuator communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator by controller (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """

        self.controller = self.ini_stage_init(old_controller=controller,
                                              new_controller=GBSAX())

        info = "Whatever info you want to log"
        initialized = True
        return info, initialized

    def move_abs(self, value):
        """ Move the actuator to the absolute target defined by value

        Parameters
        ----------
        value: (float) value of the absolute target positioning
        """

        value = self.check_bound(value)  #if user checked bounds, the defined bounds are applied here
        self.target_value = value
        value = self.set_position_with_scaling(value)  # apply scaling if the user specified one
        self.controller.set_phase_in_object_plane(value)  # when writing your own plugin replace this line
        self.emit_status(ThreadCommand('Update_Status', ['Some info you want to log']))

    def move_rel(self, value):
        """ Move the actuator to the relative target actuator value defined by value

        Parameters
        ----------
        value: (float) value of the relative target positioning
        """
        value = self.check_bound(self.current_position + value) - self.current_position
        self.target_value = value + self.current_position
        value = self.set_position_relative_with_scaling(value)

        self.controller.your_method_to_set_a_relative_value(value)  # when writing your own plugin replace this line
        self.emit_status(ThreadCommand('Update_Status', ['Some info you want to log']))

    def move_home(self):
        """Call the reference method of the controller"""
        self.controller.your_method_to_get_to_a_known_reference()  # when writing your own plugin replace this line
        self.emit_status(ThreadCommand('Update_Status', ['Some info you want to log']))

    def stop_motion(self):
        """Stop the actuator and emits move_done signal"""
        pass


if __name__ == '__main__':
    import sys
    from qtpy import QtWidgets, QtCore
    from pymodaq.control_modules.daq_move import DAQ_Move
    from pathlib import Path
    app = QtWidgets.QApplication(sys.argv)
    if config('style', 'darkstyle'):
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet())

    Form = QtWidgets.QWidget()
    prog = DAQ_Move(Form, title='GBSAX',)

    Form.show()
    prog.actuator = 'MockHolography'
    prog.init_hardware_ui()
    while not prog._initialized_state:
        QtWidgets.QApplication.processEvents()
        QtCore.QThread.msleep(1000)
    controller: GBSAX = prog.controller
    controller.load_image()

    prog.move_abs(np.angle(controller.field_object))

    sys.exit(app.exec_())

