# -*- coding: utf-8 -*-
"""
Created the 31/08/2023

@author: Sebastien Weber
"""
from abc import ABC, abstractproperty
from typing import List
from pathlib import Path
import importlib
import pkgutil
import inspect
import numpy as np
import warnings

from pymodaq.extensions.pid.utils import DataToActuatorPID, DataToExport
from pymodaq.utils.managers.modules_manager import ModulesManager
from pymodaq.utils.config import BaseConfig, USER
from pymodaq.utils.daq_utils import find_dict_in_list_from_key_val, get_entrypoints
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.plotting.data_viewers.viewer import ViewersEnum
from pymodaq.utils.parameter import Parameter


logger = set_logger(get_module_name(__file__))


class Config(BaseConfig):
    """Main class to deal with configuration values for this plugin"""
    config_template_path = Path(__file__).parent.joinpath('resources/config_template.toml')
    config_name = f"config_{__package__.split('pymodaq_plugins_')[1]}"


class DataToActuatorOpti(DataToExport):
    """ Particular case of a DataToExport adding one named parameter to indicate what kind of change should be applied
    to the actuators, absolute or relative

    Attributes
    ----------
    mode: str
        Adds an attribute called mode holding a string describing the type of change: relative or absolute

    Parameters
    ---------
    mode: str
        either 'rel' or 'abs' for a relative or absolute change of the actuator's values
    """

    def __init__(self, *args, mode='rel', **kwargs):
        if mode not in ['rel', 'abs']:
            warnings.warn('Incorrect mode for the actuators, switching to default relative mode: rel')
            mode = 'rel'
        kwargs.update({'mode': mode})
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'{super().__repr__()}: {self.mode}'


class OptimisationModelGeneric(ABC):
    optimisation_algorithm = abstractproperty()

    actuators_name: List[str] = []
    detectors_name: List[str] = []
    observables_dim: List[ViewersEnum] = []

    params = []

    def __init__(self, optimisation_controller: 'Optimisation'):
        self.optimisation_controller = optimisation_controller  # instance of the pid_controller using this model
        self.modules_manager: ModulesManager = optimisation_controller.modules_manager

        self.settings = self.optimisation_controller.settings.child('models', 'model_params')  # set of parameters
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
        names = self.optimisation_controller.settings.child('main_settings', 'detector_modules').value()['selected']
        self.data_names = []
        for name in names:
            name = name.split('//')
            self.data_names.append(name)

    def update_settings(self, param: Parameter):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        To be overwritten in child class
        """
        ...

    def ini_models(self):
        self.modules_manager.selected_actuators_name = self.actuators_name
        self.modules_manager.selected_detectors_name = self.detectors_name

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
        raise NotImplementedError

    def convert_output(self, outputs: List[np.ndarray]) -> DataToActuatorOpti:
        """
        Convert the output of the Optimisation Controller in units to be fed into the actuators
        Parameters
        ----------
        outputs: list of numpy ndarray
            output value from the controller from which the model extract a value of the same units as the actuators

        Returns
        -------
        DataToActuatorOpti: derived from DataToExport. Contains value to be fed to the actuators with a a mode
            attribute, either 'rel' for relative or 'abs' for absolute.

        """
        raise NotImplementedError


def get_optimisation_models(model_name=None):
    """
    Get PID Models as a list to instantiate Control Actuators per degree of liberty in the model

    Returns
    -------
    list: list of disct containting the name and python module of the found models
    """
    models_import = []
    discovered_models = get_entrypoints(group='pymodaq.models')
    if len(discovered_models) > 0:
        for pkg in discovered_models:
            try:
                module = importlib.import_module(pkg.value)
                module_name = pkg.value

                for mod in pkgutil.iter_modules([str(Path(module.__file__).parent.joinpath('models'))]):
                    try:
                        model_module = importlib.import_module(f'{module_name}.models.{mod.name}', module)
                        classes = inspect.getmembers(model_module, inspect.isclass)
                        for name, klass in classes:
                            if klass.__name__ in model_module.__name__:
                                models_import.append({'name': mod.name, 'module': model_module, 'class': klass})
                                break

                    except Exception as e:  # pragma: no cover
                        logger.warning(str(e))

            except Exception as e:  # pragma: no cover
                logger.warning(f'Impossible to import the {pkg.value} extension: {str(e)}')

    if model_name is None:
        return models_import
    else:
        return find_dict_in_list_from_key_val(models_import, 'name', model_name)
