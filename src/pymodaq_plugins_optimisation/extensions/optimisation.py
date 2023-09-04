from pymodaq.utils import gui_utils as gutils
from pymodaq.utils import daq_utils as utils
from pyqtgraph.parametertree import Parameter, ParameterTree
from pymodaq.utils.parameter import pymodaq_ptypes
from qtpy import QtWidgets, QtCore
import time
import numpy as np
from pymodaq.utils.data import DataToExport, DataActuator
from pymodaq.utils.plotting.data_viewers.viewer1D import Viewer1D
from pymodaq.utils.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq_plugins_optimisation.utils import get_optimisation_models, OptimisationModelGeneric
from pymodaq.utils.gui_utils import QLED

config = utils.load_config()
logger = utils.set_logger(utils.get_module_name(__file__))

EXTENSION_NAME = 'Optimisation'
CLASS_NAME = 'Optimisation'


class Optimisation(gutils.CustomApp):
    command_runner = QtCore.Signal(utils.ThreadCommand)
    models = get_optimisation_models()

    params = [
        {'title': 'Models', 'name': 'models', 'type': 'group', 'expanded': True, 'visible': True, 'children': [
            {'title': 'Models class:', 'name': 'model_class', 'type': 'list',
             'limits': [d['name'] for d in models]},
            {'title': 'Model params:', 'name': 'model_params', 'type': 'group', 'children': []},
        ]},
        {'title': 'Move settings:', 'name': 'move_settings', 'expanded': True, 'type': 'group', 'visible': False,
         'children': [
             {'title': 'Units:', 'name': 'units', 'type': 'str', 'value': ''}]},
        # here only to be compatible with DAQ_Scan, the model could update it

        {'title': 'Main Settings:', 'name': 'main_settings', 'expanded': True, 'type': 'group', 'children': []},

    ]

    def __init__(self, dockarea, dashboard):
        super().__init__(dockarea, dashboard)
        self.setup_ui()

        self.model_class: OptimisationModelGeneric = None

    def setup_docks(self):
        """
        to be subclassed to setup the docks layout
        for instance:

        self.docks['ADock'] = gutils.Dock('ADock name)
        self.dockarea.addDock(self.docks['ADock"])
        self.docks['AnotherDock'] = gutils.Dock('AnotherDock name)
        self.dockarea.addDock(self.docks['AnotherDock"], 'bottom', self.docks['ADock"])

        See Also
        ########
        pyqtgraph.dockarea.Dock
        """
        self.docks['settings'] = gutils.Dock('Settings')
        self.dockarea.addDock(self.docks['settings'])
        self.docks['settings'].addWidget(self.settings_tree)

    def setup_menu(self):
        '''
        to be subclassed
        create menu for actions contained into the self.actions_manager, for instance:

        For instance:

        file_menu = self.menubar.addMenu('File')
        self.actions_manager.affect_to('load', file_menu)
        self.actions_manager.affect_to('save', file_menu)

        file_menu.addSeparator()
        self.actions_manager.affect_to('quit', file_menu)
        '''
        pass

    def value_changed(self, param):
        ''' to be subclassed for actions to perform when one of the param's value in self.settings is changed

        For instance:
        if param.name() == 'do_something':
            if param.value():
                print('Do something')
                self.settings.child('main_settings', 'something_done').setValue(False)

        Parameters
        ----------
        param: (Parameter) the parameter whose value just changed
        '''
        pass

    def param_deleted(self, param):
        ''' to be subclassed for actions to perform when one of the param in self.settings has been deleted

        Parameters
        ----------
        param: (Parameter) the parameter that has been deleted
        '''
        raise NotImplementedError

    def child_added(self, param):
        ''' to be subclassed for actions to perform when a param  has been added in self.settings

        Parameters
        ----------
        param: (Parameter) the parameter that has been deleted
        '''
        raise NotImplementedError

    def setup_actions(self):
        logger.debug('setting actions')
        self.add_action('quit', 'Quit', 'close2', "Quit program")
        self.add_action('ini_model', 'Init Model', 'ini')
        self.add_widget('model_led', QLED, toolbar=self.toolbar)
        self.add_action('ini_runner', 'Init the Optimisation Algorithm', 'ini', checkable=True)
        self.add_widget('runner_led', QLED, toolbar=self.toolbar)
        self.add_action('run', 'Run Optimisation', 'run2', checkable=True)
        self.add_action('stop', 'Stop Optimisation', 'stop')
        self.add_action('pause', 'Pause Optimisation', 'pause')
        logger.debug('actions set')

    def connect_things(self):
        logger.debug('connecting things')
        self.connect_action('quit', self.quit, )
        self.connect_action('ini_model', self.ini_model)
        self.connect_action('ini_runner', self.ini_optimisation_runner)
        self.connect_action('run', self.run_optimisation)

    def quit(self):
        self.dockarea.parent().close()

    def set_model(self):
        model_name = self.settings.child('models', 'model_class').value()
        self.model_class = utils.find_dict_in_list_from_key_val(self.models, 'name', model_name)['class'](self)
        self.model_class.ini_model()

    def ini_model(self):
        try:
            if self.model_class is None:
                self.set_model()

            self.modules_manager.selected_actuators_name = self.model_class.actuators_name
            self.modules_manager.selected_detectors_name = self.model_class.detectors_name

            self.enable_controls_opti(True)
            self.get_action('model_led').set_as_true()
            self.set_action_enabled('ini_model', False)

        except Exception as e:
            logger.exception(str(e))

    def ini_optimisation_runner(self):
        if self.is_action_checked('ini_runner'):
            self.runner_thread = QtCore.QThread()
            runner = OptimisationRunner(self.model_class, self.modules_manager)
            self.runner_thread.runner = runner
            runner.algo_output_signal.connect(self.process_output)
            self.command_runner.connect(runner.queue_command)

            runner.moveToThread(self.runner_thread)

            self.runner_thread.start()
            self.get_action('runner_led').set_as_true()

    def process_output(self, data: DataToExport):
        pass

    def enable_controls_opti(self, enable: bool):
        pass

    def run_optimisation(self):
        if self.is_action_checked('run'):
            self.command_runner.emit(utils.ThreadCommand('start', []))
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()
            self.command_runner.emit(utils.ThreadCommand('run'))
        else:
            self.command_runner.emit(utils.ThreadCommand('stop'))

            QtWidgets.QApplication.processEvents()


class OptimisationRunner(QtCore.QObject):
    algo_output_signal = QtCore.Signal(DataToExport)

    def __init__(self, model_class, module_manager):
        super().__init__()

        self.model_class = model_class
        self.module_manager = module_manager

        self.running = True

        self.optimisation_algorithm = self.model_class.optimisation_algorithm

    @QtCore.Slot(utils.ThreadCommand)
    def queue_command(self, command: utils.ThreadCommand):
        """
        """
        if command.command == "run":
            self.run_opti(*command.attributes)

        elif command.command == "pause":
            self.pause_opti(*command.attributes)

        elif command.command == "stop":
            self.stop_opti()

    def run_opti(self, sync_detectors=True, sync_acts=False):
        """Start the optimisation loop

        Parameters
        ----------
        sync_detectors: (bool) if True will make sure all selected detectors (if any) all got their data before calling
            the model
        sync_acts: (bool) if True will make sure all selected actuators (if any) all reached their target position
         before calling the model
        """
        self.running = True
        try:
            if sync_detectors:
                self.modules_manager.connect_detectors()
            if sync_acts:
                self.modules_manager.connect_actuators()

            self.current_time = time.perf_counter()
            logger.info('PID loop starting')
            while self.running:
                # print('input: {}'.format(self.input))
                # # GRAB DATA FIRST AND WAIT ALL DETECTORS RETURNED

                self.det_done_datas = self.modules_manager.grab_datas()

                self.inputs_from_dets = self.model_class.convert_input(self.det_done_datas)

                # # EXECUTE THE optimisation
                self.outputs = []
                for ind, pid in enumerate(self.pids):
                    self.outputs.append(pid(self.inputs_from_dets.values[ind]))

                # # APPLY THE PID OUTPUT TO THE ACTUATORS
                if self.outputs is None:
                    self.outputs = [pid.setpoint for pid in self.pids]

                dt = time.perf_counter() - self.current_time
                self.outputs_to_actuators = self.model_class.convert_output(self.outputs, dt, stab=True)

                if not self.paused:
                    self.modules_manager.move_actuators(self.outputs_to_actuators.values,
                                                        self.outputs_to_actuators.mode,
                                                        polling=False)

                self.current_time = time.perf_counter()
                QtWidgets.QApplication.processEvents()
                QThread.msleep(int(self.sample_time * 1000))

            logger.info('PID loop exiting')
            self.modules_manager.connect_actuators(False)
            self.modules_manager.connect_detectors(False)

        except Exception as e:
            logger.exception(str(e))


def main(init_qt=True):
    import sys
    from pathlib import Path
    from pymodaq.utils.daq_utils import get_set_preset_path

    if init_qt:  # used for the test suite
        app = QtWidgets.QApplication(sys.argv)
        if config['style']['darkstyle']:
            import qdarkstyle
            app.setStyleSheet(qdarkstyle.load_stylesheet())

    from pymodaq.dashboard import DashBoard

    win = QtWidgets.QMainWindow()
    area = gutils.dock.DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('PyMoDAQ Dashboard')

    dashboard = DashBoard(area)
    daq_scan = None
    file = Path(get_set_preset_path()).joinpath(f"{'holography'}.xml")
    if file.exists():
        dashboard.set_preset_mode(file)
        daq_scan = dashboard.load_extension_from_name('Optimisation')
    else:
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(f"The default file specified in the configuration file does not exists!\n"
                       f"{file}\n"
                       f"Impossible to load the DAQScan Module")
        msgBox.setStandardButtons(msgBox.Ok)
        ret = msgBox.exec()

    if init_qt:
        sys.exit(app.exec_())
    return dashboard, daq_scan, win


if __name__ == '__main__':
    main()



