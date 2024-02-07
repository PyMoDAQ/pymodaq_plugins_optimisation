from typing import List, Union

from pymodaq.utils import gui_utils as gutils
from pymodaq.utils import daq_utils as utils
from pymodaq.utils.parameter import utils as putils
from qtpy import QtWidgets, QtCore
import time
import numpy as np
from pymodaq.utils.data import DataToExport, DataActuator, DataCalculated
from pymodaq.utils.plotting.data_viewers.viewer0D import Viewer0D


from pymodaq.utils.plotting.data_viewers.viewer import ViewerDispatcher
from pymodaq_plugins_optimisation.utils import get_optimisation_models, OptimisationModelGeneric, DataToActuatorOpti
from pymodaq.utils.gui_utils import QLED
from pymodaq.utils.managers.modules_manager import ModulesManager

from pymodaq.utils.config import Config
from pymodaq_plugins_optimisation import config as plugin_config
logger = utils.set_logger(utils.get_module_name(__file__))

config = Config()

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

        self.viewer_fitness: Viewer0D = None
        self.viewer_observable: ViewerDispatcher = None
        self.model_class: OptimisationModelGeneric = None

        self.setup_ui()

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

        # widget_fitness = QtWidgets.QWidget()
        # self.viewer_fitness = Viewer0D(widget_fitness)
        # self.docks['fitness'] = gutils.Dock('Fitness')
        # self.dockarea.addDock(self.docks['fitness'], 'right', self.docks['settings'])
        # self.docks['fitness'].addWidget(widget_fitness)

        widget_observable = QtWidgets.QWidget()
        widget_observable.setLayout(QtWidgets.QHBoxLayout())
        observable_dockarea = gutils.DockArea()
        widget_observable.layout().addWidget(observable_dockarea)
        self.viewer_observable = ViewerDispatcher(observable_dockarea)
        self.docks['observable'] = gutils.Dock('Observable')
        self.dockarea.addDock(self.docks['observable'], 'right', self.docks['settings'])
        self.docks['observable'].addWidget(widget_observable)

        if len(self.models) != 0:
            self.get_set_model_params(self.models[0]['name'])

    def get_set_model_params(self, model_name):
        self.settings.child('models', 'model_params').clearChildren()
        if len(self.models) > 0:
            model_class = utils.find_dict_in_list_from_key_val(self.models, 'name', model_name)['class']
            params = getattr(model_class, 'params')
            self.settings.child('models', 'model_params').addChildren(params)

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
        if param.name() == 'model_class':
            self.get_set_model_params(param.value())
        elif param.name() in putils.iter_children(self.settings.child('models', 'model_params'), []):
            if self.model_class is not None:
                self.model_class.update_settings(param)

    def setup_actions(self):
        logger.debug('setting actions')
        self.add_action('quit', 'Quit', 'close2', "Quit program")
        self.add_action('ini_model', 'Init Model', 'ini')
        self.add_widget('model_led', QLED, toolbar=self.toolbar)
        self.add_action('ini_runner', 'Init the Optimisation Algorithm', 'ini', checkable=True)
        self.add_widget('runner_led', QLED, toolbar=self.toolbar)
        self.add_action('run', 'Run Optimisation', 'run2', checkable=True)
        self.add_action('pause', 'Pause Optimisation', 'pause', checkable=True)
        logger.debug('actions set')

    def connect_things(self):
        logger.debug('connecting things')
        self.connect_action('quit', self.quit, )
        self.connect_action('ini_model', self.ini_model)
        self.connect_action('ini_runner', self.ini_optimisation_runner)
        self.connect_action('run', self.run_optimisation)
        self.connect_action('pause', self.pause_runner)

    def pause_runner(self):
        self.command_runner.emit(utils.ThreadCommand('pause_PID', self.is_action_checked('pause')))

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

            self.viewer_observable.update_viewers(['Viewer0D'] + self.model_class.observables_dim,
                                                  ['Fitness', 'Observable', 'Individual'])

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
        # fitness
        # self.viewer_fitness.show_data(data.get_data_from_name('fitness'))
        # data.remove()
        self.viewer_observable.show_data(data)

    def enable_controls_opti(self, enable: bool):
        pass

    def run_optimisation(self):
        if self.is_action_checked('run'):
            self.get_action('run').set_icon('stop')
            self.command_runner.emit(utils.ThreadCommand('start', {}))
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()
            self.command_runner.emit(utils.ThreadCommand('run', {}))
        else:
            self.get_action('run').set_icon('run2')
            self.command_runner.emit(utils.ThreadCommand('stop', {}))

            QtWidgets.QApplication.processEvents()


class OptimisationRunner(QtCore.QObject):
    algo_output_signal = QtCore.Signal(DataToExport)

    def __init__(self, model_class: OptimisationModelGeneric, modules_manager: ModulesManager):
        super().__init__()

        self.det_done_datas: DataToExport = None
        self.inputs_from_dets: DataToExport = None
        self.outputs: List[np.ndarray] = []
        self.dte_actuators: DataToExport = None

        self.model_class: OptimisationModelGeneric = model_class
        self.modules_manager: ModulesManager = modules_manager

        self.running = True
        self.paused = False

        self.optimisation_algorithm = self.model_class.optimisation_algorithm

    @QtCore.Slot(utils.ThreadCommand)
    def queue_command(self, command: utils.ThreadCommand):
        """
        """
        if command.command == "run":
            self.run_opti(**command.attribute)

        elif command.command == "pause":
            self.pause_opti(command.attribute)

        elif command.command == "stop":
            self.running = False

    def pause_opti(self, pause_state: bool):
        # for ind, pid in enumerate(self.pids):
        #     if pause_state:
        #         pid.set_auto_mode(False)
        #         logger.info('Stabilization paused')
        #     else:
        #         pid.set_auto_mode(True, self.outputs[ind])
        #         logger.info('Stabilization restarted from pause')
        self.paused = pause_state

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
            logger.info('Optimisation loop starting')
            while self.running:
                # print('input: {}'.format(self.input))
                # # GRAB DATA FIRST AND WAIT ALL DETECTORS RETURNED

                self.det_done_datas = self.modules_manager.grab_datas()

                self.inputs_from_dets = self.model_class.convert_input(self.det_done_datas)

                # # EXECUTE THE optimisation
                self.outputs: List[np.ndarray] = []
                self.outputs = [self.optimisation_algorithm.evolve(self.inputs_from_dets)]

                dte = DataToExport('algo',
                                   data=[DataCalculated('fitness',
                                                        data=[np.array([self.optimisation_algorithm.fitness])]),
                                         ])

                # # # APPLY THE population OUTPUT TO THE ACTUATOR
                # if self.outputs is None:
                #     self.outputs = [pid.setpoint for pid in self.pids]

                dt = time.perf_counter() - self.current_time
                self.output_to_actuators: DataToActuatorOpti = self.model_class.convert_output(self.outputs)

                dte.append(self.inputs_from_dets)
                dte.append(self.output_to_actuators)
                self.algo_output_signal.emit(dte)

                if not self.paused:
                    self.modules_manager.move_actuators(self.output_to_actuators,
                                                        self.output_to_actuators.mode,
                                                        polling=False)

                self.current_time = time.perf_counter()
                QtWidgets.QApplication.processEvents()
                #QtCore.QThread.msleep(int(self.sample_time * 1000))

            logger.info('Optimisation loop exiting')
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
        if config('style', 'darkstyle'):
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



