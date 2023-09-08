# -*- coding: utf-8 -*-
"""
Created the 08/09/2023

@author: Sebastien Weber
"""
from typing import List, Union, Tuple
from pathlib import Path

import numpy as np

from pymodaq_plugins_optimisation.utils import OptimisationModelGeneric
from pymodaq_plugins_optimisation.hardware.gershberg_saxton import GBSAX
from pymodaq.utils.managers.modules_manager import ModulesManager
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataToExport, DataActuator, DataWithAxes
from pymodaq.utils import math_utils as mutils
from pymodaq.extensions.pid.utils import DataToActuatorPID
from skimage.io import imread
from skimage.color import rgb2gray

from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.termination import get_termination
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling

cheshire_cat_path = Path(__file__).parent.parent.joinpath('resources/cheshirecat_rect.png')
logger = set_logger(get_module_name(__file__))


try:
    import pyfftw
    pyfftw.interfaces.cache.enable()


    def wrap_fft(*args, **kwargs):
        fft2 = pyfftw.interfaces.numpy_fft.fft2(threads=8, *args, **kwargs)
        return fft2


    def wrap_ifft(*args, **kwargs):
        ifft2 = pyfftw.interfaces.numpy_fft.ifft2(threads=8, *args, **kwargs)
        return ifft2


    def wrap_fftshift(*args, **kwargs):
        fftshift = pyfftw.interfaces.numpy_fft.fftshift(*args, **kwargs)
        return fftshift


    def wrap_ifftshift(*args, **kwargs):
        ifftshift = pyfftw.interfaces.numpy_fft.ifftshift(*args, **kwargs)
        return ifftshift


    fft2 = wrap_fft
    ifft2 = wrap_ifft

    fftshift = wrap_fftshift
    ifftshift = wrap_ifftshift

except ImportError:
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
    fftshift = np.fft.fftshift
    ifftshift = np.fft.ifftshift
    print("Warning: using numpy FFT implementation.  Consider using pyFFTW for faster Fourier transforms.")


class InputIntensity:
    def __init__(self, npixels=(768, 1024), size_pixel=0.036, size=(11, 11)):
        self.size_pixel = size_pixel  # pixel size of SLM in mm
        self.size_x = size[1]  # x-axis intensity beam size in mm (FWHM)
        self.size_y = size[0]  # y-axis intensity beam size in mm (FWHM)

        self.npixels = npixels
        x = np.arange(0, npixels[1], 1)
        y = np.arange(0, npixels[0], 1)

        #   ===   Amplitude   ===============================================
        self._amp = np.sqrt(mutils.gauss2D(x, npixels[1] / 2, self.size_x / size_pixel,
                                           y, npixels[0] / 2, self.size_y / size_pixel))

    @property
    def amplitude(self):
        return self._amp

    @property
    def intensity(self):
        return np.power(self._amp, 2.)

    def normalise_to_intensity(self, data_int: np.ndarray):
        """ Normalise an intensity like 2D array to this input total intensity

        Parameters
        ----------
        data_int: ndarray
            the array to be normalised with respect to the total intensity

        Returns
        -------
        ndarray
        """
        return data_int * np.sum(self.intensity) / np.sum(data_int)


class TargetIntensityMatchProblem(Problem):
    def __init__(self, target_intensity: Union[np.ndarray, str, Path], input_laser_size=(11, 11)):

        self.target_intensity: np.ndarray = None
        self.load_target(target_intensity)
        self.object_shape = self.target_intensity.shape

        super().__init__(n_var=np.prod(self.object_shape), n_obj=1, xl=0., xu=2 * np.pi)

        self.input_intensity = InputIntensity(npixels=self.object_shape, size=input_laser_size)

    def load_target(self, target_intensity: Union[np.ndarray, str, Path]):
        if isinstance(target_intensity, str) or isinstance(target_intensity, Path):
            img = imread(target_intensity)
            if len(img.shape) == 2:
                self.target_intensity = img
            elif len(img.shape) == 3:
                self.target_intensity = rgb2gray(img[..., 0:3])
        elif isinstance(target_intensity, np.ndarray):
            self.target_intensity = target_intensity
        else:
            raise TypeError('Invalid type for the target intensity')

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """

        Parameters
        ----------
        x: np.ndarray
            ndarray of shape (npop, n_var)
        out: dict
            dictionary where fitness values shoudl be written in out['F'] while constraints sould be (if any) written
            in out['G']
        args
        kwargs

        Returns
        -------

        """
        reshape = [x.shape[0]]
        reshape.extend(list(self.target_intensity.shape))
        reshape = tuple(reshape)
        out['F'] = self.fitness(x.reshape(reshape))

    def fitness(self, images: np.ndarray) -> np.ndarray:
        """ Compute the fitness of a bunch of images

        Parameters
        ----------
        images: np.ndarray
            ndarray of shape (npop, self.target_intensity.shape)
        Returns
        -------
        ndarray of shape (npop)
        """
        return 100 * np.sum(np.abs(np.subtract(images, np.sqrt(self.target_intensity))) ** 2, axis=(-2, -1)) \
               / np.prod(images.shape) / np.sum(self.target_intensity)

    def get_name(self):
        return "Try to match a given image loaded as a numpy array"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.n_var)


class GbSaxPymooAlgorithm(Algorithm):
    def __init__(self,
                 pop_size=1,
                 sampling=FloatRandomSampling(),
                 **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size

        self.initialization = Initialization(sampling)

    def _setup(self, problem, **kwargs):
        pass

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        pass

    def _infill(self):
        pass

    def _advance(self, infills=None, **kwargs):
        pass

    def _finalize(self):
        pass


if __name__ == '__main__':
    problem = TargetIntensityMatchProblem(cheshire_cat_path)
    algo = GbSaxPymooAlgorithm(pop_size=1)
    algo.setup(problem, termination=('n_gen', 10), verbose=True)

    pop = algo.ask()
    algo.evaluator.eval(problem, pop)


# class GBSAX:
#
#     def __init__(self):
#
#         self.object_shape = (768, 1024)
#         self.image_shape: Tuple[int, int] = None
#
#         self.target_intensity: np.ndarray = None
#
#
#         self.input_intensity = InputIntensity(npixels=self.object_shape, size=self.input_size)
#
#         self.field_object: np.ndarray = None
#         self.field_image: np.nd_array = None
#         self.field_object_phase: np.ndarray = None
#         self._sse: float = 100
#
#     @property
#     def input_size(self):
#         return self._input_size
#
#     @input_size.setter
#     def input_size(self, input_size: Tuple[float, float]):
#         if len(input_size) == 2:
#             self._input_size = input_size
#             self.input_intensity = InputIntensity(npixels=self.object_shape, size=self._input_size)
#
#     def load_image(self, fname: Union[str, Path] = cheshire_cat_path):
#         img = imread(fname)
#         if len(img.shape) == 2:
#             self.target_intensity = img
#         elif len(img.shape) == 3:
#             self.target_intensity = rgb2gray(img[..., 0:3])
#
#         self.image_shape = self.target_intensity.shape
#         self.target_intensity = self.input_intensity.normalise_to_intensity(self.target_intensity)
#
#         self.set_phase_in_object_plane((np.random.rand(*self.object_shape) - 0.5) * 2 * np.pi)
#         self.propagate_field()
#
#     def set_phase_in_object_plane(self, phase: np.ndarray):
#         self.field_object_phase = phase
#         if phase.shape == self.object_shape:
#             self.field_object = self.input_intensity.amplitude * np.exp(1j * phase)
#             self.propagate_field()
#         else:
#             raise ValueError('The phase shape is incoherent with the parameters')
#
#     def get_phase(self):
#         return self.field_object_phase
#
#     def get_npad(self):
#         npad = np.abs(np.array(self.object_shape) - np.array(self.image_shape)) / 2
#         npad = tuple(np.asarray(npad, int))
#
#         return npad
#
#     def propagate_field(self):
#         npad = self.get_npad()
#         field_object_padded = np.pad(self.field_object, ((npad[0], npad[0]), (npad[1], npad[1])),
#                                      constant_values=(0, 0))
#
#         self.field_image = fftshift(fft2(fftshift(field_object_padded))) / \
#                            np.sqrt(np.prod(field_object_padded.shape))
#
#     def evolve_field(self) -> np.ndarray:
#         npad = self.get_npad()
#         self._sse = 100 * np.sum(np.abs(np.sqrt(self.target_intensity) - self.field_image)) ** 2 \
#                     / np.prod(self.image_shape) / np.sum(self.target_intensity)
#         field_image_corrected = np.sqrt(self.target_intensity) * np.exp(1j * np.angle(self.field_image))
#         field_object_corrected = ifftshift(ifft2(ifftshift(field_image_corrected)))
#         field_object_corrected = field_object_corrected[npad[0] + 1:npad[0] + 1 + self.object_shape[0],
#                                  npad[1] + 1:npad[1] + 1 + self.object_shape[1]]
#         self.field_object_phase = np.angle(field_object_corrected)
#         return self.field_object_phase
#
#     @property
#     def fitness(self) -> float:
#         return self._sse
#
#     def grab(self):
#         self.propagate_field()
#         return self.field_image
#
#     def evolve(self, input: Population = None) -> Population:
#         phase = self.evolve_field()
#         return
#
#     @property
#     def sse(self):
#         return self._sse
#
#     @property
#     def intensity_image(self):
#         return np.abs(self.field_image) ** 2
#

# class OptimisationModelHolography(OptimisationModelGeneric):
#
#     optimisation_algorithm = GBSAX()
#
#     actuators_name = ["Shaper"]
#     detectors_name = ['Camera']
#
#     params = []
#
#     def __init__(self, optimisation_controller):
#         self.opti_controller = optimisation_controller  # instance of the pid_controller using this model
#         self.modules_manager: ModulesManager = optimisation_controller.modules_manager
#
#         self.settings = self.opti_controller.settings.child('models', 'model_params')  # set of parameters
#         self.check_modules(self.modules_manager)
#
#     def check_modules(self, modules_manager):
#         for act in self.actuators_name:
#             if act not in modules_manager.actuators_name:
#                 logger.warning(f'The actuator {act} defined in the PID model is not present in the Dashboard')
#                 return False
#         for det in self.detectors_name:
#             if det not in modules_manager.detectors_name:
#                 logger.warning(f'The detector {det} defined in the PID model is not present in the Dashboard')
#
#     def update_detector_names(self):
#         names = self.opti_controller.settings.child('main_settings', 'detector_modules').value()['selected']
#         self.data_names = []
#         for name in names:
#             name = name.split('//')
#             self.data_names.append(name)
#
#     def update_settings(self, param):
#         """
#         Get a parameter instance whose value has been modified by a user on the UI
#         To be overwritten in child class
#         """
#         if param.name() == '':
#             pass
#
#     def ini_model(self):
#         self.modules_manager.selected_actuators_name = self.actuators_name
#         self.modules_manager.selected_detectors_name = self.detectors_name
#         self.optimisation_algorithm: GBSAX = self.modules_manager.actuators[0].controller
#         self.optimisation_algorithm.load_image()
#
#     def convert_input(self, measurements: DataToExport):
#         """
#         Convert the measurements in the units to be fed to the Optimisation Controller
#         Parameters
#         ----------
#         measurements: DataToExport
#             data object exported from the detectors from which the model extract a value of the same units as
#             the setpoint
#
#         Returns
#         -------
#         float: the converted input
#
#         """
#         return DataToExport('inputs', data=[measurements.get_data_from_name('GBSAX Intensity')])
#
#     def convert_output(self, outputs: List[np.ndarray]) -> DataToActuatorPID:
#         """
#         Convert the output of the Optimisation Controller in units to be fed into the actuators
#         Parameters
#         ----------
#         outputs: list of numpy ndarray
#             output value from the controller from which the model extract a value of the same units as the actuators
#
#         Returns
#         -------
#         DataToActuatorPID: derived from DataToExport. Contains value to be fed to the actuators with a a mode
#             attribute, either 'rel' for relative or 'abs' for absolute.
#
#         """
#         return DataToActuatorPID('outputs', mode='abs', data=[DataActuator(self.actuators_name[ind], outputs[ind])
#                                                               for ind in range(len(outputs))])
#
#     def update_settings(self, param):
#         """
#         Get a parameter instance whose value has been modified by a user on the UI
#         Parameters
#         ----------
#         param: (Parameter) instance of Parameter object
#         """
#         if param.name() == '':
#             pass


if __name__ == '__main__':
    pass
