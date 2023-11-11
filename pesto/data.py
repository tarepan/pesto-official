import torch
from torch import Tensor, nn

from .cqt import CQT


class DataProcessor(nn.Module):

    def __init__(self,
                 step_size:         float,
                 bins_per_semitone: int          = 3,
                 device:            torch.device = torch.device("cpu"),
                 **cqt_kwargs):
        """
        NOTE: Instantiated only through `load_dataprocessor` in PESTO.
              In this usage, args are 'step_size=step_size, device=device, **cqt_args'

        Args:
            step_size         - CQT hop size [sec]
            bins_per_semitone - 
            device            - 
        """
        super().__init__()

        # Parameters
        self.step_size, self.bins_per_semitone, self.device = step_size, bins_per_semitone, device
        ## CQT-related stuff
        self.cqt_kwargs = cqt_kwargs
        self.cqt_kwargs["bins_per_octave"] = 12 * bins_per_semitone
        ## log-magnitude
        self.eps = torch.finfo(torch.float32).eps
        ## cropping
        self.lowest_bin  = int(11 * self.bins_per_semitone / 2 + 0.5)
        self.highest_bin = self.lowest_bin + 88 * self.bins_per_semitone

        # States
        self._sampling_rate = None # Should be set before first `forward` call through `sampling_rate` setter
        self.cqt            = None # Automatically initialized through `sampling_rate` setter

    def forward(self, x: Tensor):
        r"""

        Args:
            x :: (T,)               - audio waveform, any sampling rate. NOTE: shape seems to be wrong, correctly, `(T,) | (B, T)`
        Returns:
              :: (B=b*frm, 1, Freq) - log-magnitude CQT
        """

        # CQT :: (T,) | (B, T) | (B, C=1, T) -> (B, Freq, Frame, ReIm=2) -> (B, Freq, Frame) -> (B, Frame, Freq)
        complex_cqt = torch.view_as_complex(self.cqt(x)).transpose(1, 2)

        # reshape and crop borders to fit training input shape :: (B, Frame, Freq=f) -> (B, Frame, Freq=f')
        complex_cqt = complex_cqt[..., self.lowest_bin: self.highest_bin]

        # flatten eventual batch dimensions so that batched audios can be processed in parallel :: (B=b, Frame=frm, Freq) -> (B=b*frm, Freq) -> (B=b*frm, 1, Freq)
        complex_cqt = complex_cqt.flatten(0, 1).unsqueeze(1)

        # convert to dB
        log_cqt = complex_cqt.abs().clamp_(min=self.eps).log10_().mul_(20)

        return log_cqt

    def _init_cqt_layer(self, sr: int, hop_length: int):
        # NOTE: In PESTO, always `output_format="Complex"` (forward can change output_format, but PESTO do not use the argument.)
        self.cqt = CQT(sr=sr, hop_length=hop_length, **self.cqt_kwargs).to(self.device)

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sr: int):
        if sr == self._sampling_rate:
            return

        hop_length = int(self.step_size * sr + 0.5)
        self._init_cqt_layer(sr, hop_length)
        self._sampling_rate = sr
