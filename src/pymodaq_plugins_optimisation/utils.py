# -*- coding: utf-8 -*-
"""
Created the 31/08/2023

@author: Sebastien Weber
"""
from pathlib import Path
import importlib
import pkgutil
import inspect

from pymodaq.utils.config import BaseConfig, USER
from pymodaq.utils.daq_utils import find_dict_in_list_from_key_val, get_entrypoints
from pymodaq.utils.logger import set_logger, get_module_name


logger = set_logger(get_module_name(__file__))


class Config(BaseConfig):
    """Main class to deal with configuration values for this plugin"""
    config_template_path = Path(__file__).parent.joinpath('resources/config_template.toml')
    config_name = f"config_{__package__.split('pymodaq_plugins_')[1]}"


class OptimisationModelGeneric:
    pass


def get_optimisation_models(model_name=None):
    """
    Get PID Models as a list to instantiate Control Actuators per degree of liberty in the model

    Returns
    -------
    list: list of disct containting the name and python module of the found models
    """
    models_import = []
    discovered_models = get_entrypoints(group='pymodaq.optimisation_models')
    if len(discovered_models) > 0:
        for pkg in discovered_models:
            try:
                module = importlib.import_module(pkg.value)
                module_name = pkg.value

                for mod in pkgutil.iter_modules([str(Path(module.__file__).parent.joinpath('optimisation_models'))]):
                    try:
                        model_module = importlib.import_module(f'{module_name}.optimisation_models.{mod.name}', module)
                        classes = inspect.getmembers(model_module, inspect.isclass)
                        for name, klass in classes:
                            if klass.__base__ is OptimisationModelGeneric:
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
