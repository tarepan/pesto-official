r"""The implementation of the CQT comes from the nnAudio repository: https://github.com/KinWaiCheuk/nnAudio
Due to conflicts between some versions of NumPy and nnAudio, we use the implementation as is instead of adding nnAudio
to the requirements of this project. Compared to the original implementation, some minor modifications have been done
in the code, however the behaviour remains the same.
"""
from typing import Optional

import numpy as np
import warnings
from scipy.signal import get_window

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def broadcast_dim(x):
    """Auto broadcast input so that it can fits into a Conv1d.
    
    Args:
        x :: (T,) | (B=b, T) | (B=b, C=c, T) - Input
    Returns:
          :: (B=1|b, C=1|c, T)               - Broadcasted output
    """

    # :: (B=b, T) -> (B=b, C=1, T)
    if x.dim() == 2:
        x = x[:, None, :]
    # :: (T,) -> (B=1, C=1, T)
    elif x.dim() == 1:
        x = x[None, None, :]
    # :: (B=b, C=c, T) -> (B=b, C=c, T)
    elif x.dim() == 3:
        pass
    else:
        raise ValueError("Only support input with shape = (batch, len) or shape = (len)")

    return x


def nextpow2(A):
    """A helper function to calculate the next nearest number to the power of 2.

    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2

    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``

    Examples
    --------

    >>> nextpow2(6)
    3
    """

    return int(np.ceil(np.log2(A)))


def get_window_dispatch(window, N, fftbins=True):
    if isinstance(window, str):
        return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning(
            "You are using Kaiser window with beta factor "
            + str(window)
            + ". Correct behaviour not checked."
        )
    else:
        raise Exception(
            "The function get_window from scipy only supports strings, tuples and floats."
        )


def create_cqt_kernels(
        Q,
        fs,
        fmin: float,
        n_bins=84,
        bins_per_octave=12,
        norm=1,
        window="hann",
        fmax: Optional[float] = None,
        topbin_check=True,
        gamma=0,
        pad_fft=True
):
    """
    Automatically create CQT kernels in time domain
    """
    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, \
                          please reduce the n_bins".format(
                np.max(freqs)
            )
        )

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = np.ceil(Q * fs / (freqs + gamma / alpha))

    # get max window length depending on gamma value
    max_len = int(max(lengths))
    fftLen = int(2 ** (np.ceil(np.log2(max_len))))

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = lengths[k]

        # Centering the kernels
        if l % 2 == 1:  # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))

        window_dispatch = get_window_dispatch(window, int(l), fftbins=True)
        sig = window_dispatch * np.exp(np.r_[-l // 2: l // 2] * 1j * 2 * np.pi * freq / fs) / l

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start: start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start: start + int(l)] = sig
        # specKernel[k, :] = fft(tempKernel[k])

    # return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()
    return tempKernel, fftLen, torch.tensor(lengths).float(), freqs


class CQT(nn.Module):
    """Calculate CQT.

    Most of the arguments follow the convention from librosa.

    This alogrithm uses the method proposed in [1].
    I slightly modify it so that it runs faster than the original 1992 algorithm, that is why I call it version 2.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a constant Q transform.” (1992).

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    filter_scale : float > 0
        Filter scale factor. Values of filter_scale smaller than 1 can be used to improve the time resolution at the
        cost of degrading the frequency resolution. Important to note is that setting for example filter_scale = 0.5 and
        bins_per_octave = 48 leads to exactly the same time-frequency resolution trade-off as setting filter_scale = 1
        and bins_per_octave = 24, but the former contains twice more frequency bins per octave. In this sense, values
        filter_scale < 1 can be seen to implement oversampling of the frequency axis, analogously to the use of zero
        padding when calculating the DFT.

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : string, float, or tuple
        The windowing function for CQT. If it is a string, It uses ``scipy.signal.get_window``. If it is a
        tuple, only the gaussian window wanrantees constant Q factor. Gaussian window should be given as a
        tuple ('gaussian', att) where att is the attenuation in the border given in dB.
        Please refer to scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``.

    Examples
    --------
    >>> spec_layer = CQT()
    >>> specs = spec_layer(x)
    """

    def __init__(
            self,
            sr=22050,
            hop_length=512,
            fmin=32.70,
            fmax=None,
            n_bins=84,
            bins_per_octave=12,
            filter_scale=1,
            norm=1,
            window="hann",
            center=True,
            pad_mode="reflect",
            trainable=False,
            output_format="Magnitude"
    ):

        super().__init__()

        self.trainable = trainable
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.output_format = output_format

        # creating kernels for CQT
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)

        self.register_buffer("lenghts", lenghts)
        self.frequencies = freqs

        cqt_kernels_real = torch.tensor(cqt_kernels.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag).unsqueeze(1)
        if trainable:  # NOTE: can't it be factorized?
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

    def forward(self, x, output_format=None, normalization_type: str = "librosa") -> Tensor:
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x :: (T,) | (B, T) | (B, C=1, T) - Waveforms
        normalization_type               - Type of the normalization. The possible options are: \n
            'librosa'       : the output fits the librosa one \n
            'convolutional' : the output conserves the convolutional inequalities of the wavelet transform:\n
            for all p ϵ [1, inf] \n
                - || CQT ||_p <= || f ||_p || g ||_1 \n
                - || CQT ||_p <= || f ||_1 || g ||_p \n
                - || CQT ||_2  = || f ||_2 || g ||_2 \n
            'wrap'          : wraps positive and negative frequencies into positive frequencies. This means that the CQT of a
            sinus (or a cosine) with a constant amplitude equal to 1 will have the value 1 in the bin corresponding to its frequency.
        Returns
        -------
          :: (B, Freq, Frame) | (B, Freq, Frame, ReIm=2) - Spectrogram, shape is former if `output_format=='Magnitude'`, later if `output_format=='Complex'` or `=='Phase'`
        
        

        """

        output_format = output_format or self.output_format

        # Reshape :: (T,) | (B, T) | (B, C=1, T) -> (B, C=1, T) - As 3D input
        x = broadcast_dim(x)

        # Centering
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == "reflect":
                padding = nn.ReflectionPad1d(self.kernel_width // 2)
            x = padding(x)

        # CQT :: (B, C=1, T) -> (B, C, Frame)
        CQT_real =  F.conv1d(x, self.cqt_kernels_real, stride=self.hop_length) # bias=None, padding=0
        CQT_imag = -F.conv1d(x, self.cqt_kernels_imag, stride=self.hop_length) # bias=None, padding=0

        # Normalization
        if normalization_type == "librosa":
            CQT_real *= torch.sqrt(self.lenghts.view(-1, 1))
            CQT_imag *= torch.sqrt(self.lenghts.view(-1, 1))
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT_real *= 2
            CQT_imag *= 2
        else:
            raise ValueError(f"The normalization_type {normalization_type} is not part of our current options.")

        if output_format == "Magnitude":
            margin = 1e-8 if self.trainable else 0
            return torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2) + margin)

        elif output_format == "Complex":
            return torch.stack((CQT_real, CQT_imag), -1)

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT_imag, CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag, CQT_real))
            return torch.stack((phase_real, phase_imag), -1)
