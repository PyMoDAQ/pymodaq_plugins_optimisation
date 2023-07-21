# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins
from pymodaq.utils import math_utils as mutils

logger = set_logger(get_module_name(__file__))

cheshire_cat_path = Path(__file__).parent.parent.joinpath('resources/cheshirecat_rect.png')

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


class GBSAX:

    def __init__(self, input_size=(11, 11)):

        self.object_shape = (768, 1024)
        self.image_shape: Tuple[int, int] = None

        self.target_intensity: np.ndarray = None

        self._input_size = input_size  # in mm
        self.input_intensity = InputIntensity(npixels=self.object_shape, size=self.input_size)

        self.field_object: np.ndarray = None
        self.field_image: np.nd_array = None
        self._sse: float = 100

    def evolve(self):
        ...

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, input_size: Tuple[float, float]):
        if len(input_size) == 2:
            self._input_size = input_size
            self.input_intensity = InputIntensity(npixels=self.object_shape, size=self._input_size)

    def load_image(self, fname: Union[str, Path] = cheshire_cat_path):
        img = imread(fname)
        if len(img.shape) == 2:
            self.target_intensity = img
        elif len(img.shape) == 3:
            self.target_intensity = rgb2gray(img[..., 0:3])

        self.image_shape = self.target_intensity.shape
        self.target_intensity = self.input_intensity.normalise_to_intensity(self.target_intensity)

        self.set_phase_in_object_plane((np.random.rand(*self.object_shape) - 0.5) * 2 * np.pi)
        self.propagate_field()

    def set_phase_in_object_plane(self, phase: np.ndarray):
        if phase.shape == self.object_shape:
            self.field_object = self.input_intensity.amplitude * np.exp(1j * phase)

        else:
            raise ValueError('The phase shape is incoherent with the parameters')

    def get_npad(self):
        npad = np.abs(np.array(self.object_shape) - np.array(self.image_shape)) / 2
        npad = tuple(np.asarray(npad, int))

        return npad

    def propagate_field(self):
        npad = self.get_npad()
        field_object_padded = np.pad(self.field_object, ((npad[0], npad[0]), (npad[1], npad[1])),
                                     constant_values=(0, 0))

        self.field_image = fftshift(fft2(fftshift(field_object_padded))) / \
                           np.sqrt(np.prod(field_object_padded.shape))

    def evolve_field(self):
        npad = self.get_npad()
        self._sse = 100 * np.sum(np.abs(np.sqrt(self.target_intensity) - self.field_image)) ** 2 \
                    / np.prod(self.image_shape) / np.sum(self.target_intensity)
        field_image_corrected = np.sqrt(self.target_intensity) * np.exp(1j * np.angle(self.field_image))
        field_object_corrected = ifftshift(ifft2(ifftshift(field_image_corrected)))

        self.set_phase_in_object_plane(np.angle(field_object_corrected[npad[0]+1:npad[0]+1+self.object_shape[0],
                                                                       npad[1]+1:npad[1]+1+self.object_shape[1]]))

    @property
    def sse(self):
        return self._sse

    @property
    def intensity_image(self):
        return np.abs(self.field_image) ** 2


if __name__ == '__main__':
    import sys
    from qtpy import QtWidgets, QtCore, QtGui
    from pymodaq.utils.config import Config
    from pymodaq.utils.plotting.data_viewers.viewer2D import Viewer2D
    from pymodaq.utils.gui_utils.widgets import PushButtonIcon, LabelWithFont
    config = Config()

    app = QtWidgets.QApplication(sys.argv)
    if config('style', 'darkstyle'):
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qdarkstyle.DarkPalette))

    widget_main = QtWidgets.QWidget()
    widget_main.setLayout(QtWidgets.QVBoxLayout())
    widget_sett = QtWidgets.QWidget()
    widget_sett.setLayout(QtWidgets.QHBoxLayout())
    pb = QtWidgets.QPushButton('Evolve me 1 !')
    run = QtWidgets.QPushButton('Evolve me 100!')
    sse = LabelWithFont(f'SSE = {100}%', font_name="Tahoma", font_size=14, isbold=True, isitalic=True)
    sse.setFont(QtGui.QFont())
    widget_sett.layout().addWidget(pb)
    widget_sett.layout().addWidget(run)
    widget_sett.layout().addWidget(sse)
    widget_main.layout().addWidget(widget_sett)
    widget = QtWidgets.QWidget()
    widget.setLayout(QtWidgets.QHBoxLayout())

    widget_main.layout().addWidget(widget)

    object_widget = QtWidgets.QWidget()
    object_viewer = Viewer2D(object_widget)

    image_widget = QtWidgets.QWidget()
    image_viewer = Viewer2D(image_widget)

    widget.layout().addWidget(object_widget)
    widget.layout().addWidget(image_widget)

    widget_main.show()
    gbsax = GBSAX()
    gbsax.load_image()

    image_viewer.show_data(DataFromPlugins('GB', data=[gbsax.target_intensity, np.abs(gbsax.field_image)**2]))
    object_viewer.show_data(DataFromPlugins('GB', data=[np.angle(gbsax.field_object)]))

    def evolve_and_plot():
        gbsax.propagate_field()
        gbsax.evolve_field()
        image_viewer.show_data(DataFromPlugins('GB', data=[gbsax.target_intensity, np.abs(gbsax.field_image)**2]))
        object_viewer.show_data(DataFromPlugins('GB', data=[np.angle(gbsax.field_object), np.abs(gbsax.field_object)]))
        sse.setText(f'SSE = {gbsax.sse}%')

    def evolve_loop():
        for _ in range(100):
            evolve_and_plot()
            QtWidgets.QApplication.processEvents()

    pb.clicked.connect(evolve_and_plot)
    run.clicked.connect(evolve_loop)

    sys.exit(app.exec_())

