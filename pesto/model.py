from functools import partial

import torch
from torch import Tensor, nn


class ToeplitzLinear(nn.Conv1d):
    """ToeplitzLinear, implemented by c1c1 Conv1d."""
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features  - Frequency dimension size of input
            out_channels - Frequency dimension size of output
        """

        super(ToeplitzLinear, self).__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features+out_features-1,
            padding=out_features-1,
            bias=False
        )

    def forward(self, input: Tensor) -> Tensor:
        """ :: (B, Freq=i) -> (B, 1, Freq=i) -> (B, 1, Freq=o) -> (B, Freq=o)"""
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)


class PESTOEncoder(nn.Module):
    """
    Basic CNN similar to the one in Johannes Zeitler's report, but for longer HCQT input
    Still with 75 (-1) context frames, i.e. 37 frames padded to each side
    """

    def __init__(
            self,
            n_chan_layers=(20, 20, 10, 1),
            n_prefilt_layers=1,
            residual=False,
            n_bins_in=216,
            output_dim=128,
            num_output_layers: int = 1
    ):
        """
        Args (Defaults: BasicCNN by Johannes Zeitler but with 1 input channel):
            n_chan_layers     - Feature dimension sizes of hidden layers. In official PESTO paper, `(40, 30, 30, 10, 3)`
            n_prefilt_layers  - The number of prefiltering layer.         In official PESTO paper, `2`
            residual          - Deprecated.
            n_bins_in         - Feature dimension size of input.          In official PESTO paper,  `88*bins_per_semitone`==` 88*3`
            output_dim        - Feature dimension size of output.         In official PESTO paper, `128*bins_per_semitone`==`128*3`
            num_output_layers - pre_fc's parameter
        
        NOTE: n_bins_in is (12 * number of octaves), n_bins_out is (12 for pitch class, 72 for pitch, num_octaves * 12)
        """
        super().__init__()

        assert residual, "Support only 'residual' mode."

        activation_layer = partial(nn.LeakyReLU, negative_slope=0.3)

        # Feature dimension size of hidden layers : tuple[int, int, int, int, int] = (40, 30, 30, 10, 3) in official PESTO
        n_ch = n_chan_layers
        if len(n_ch) < 5:
            n_ch.append(1)

        # Norm - LN
        self.layernorm = nn.LayerNorm(normalized_shape=[1, n_bins_in])
        # Prefiltering
        ## conv1 - Conv1d_k15s1same-LReLU
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=n_ch[0], kernel_size=15, padding=7, stride=1), activation_layer())
        ## prefilt_list - [Conv1d_k15s1same-LReLU]x1
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_list = nn.ModuleList()
        for p in range(1, n_prefilt_layers):
            self.prefilt_list.append(nn.Sequential(nn.Conv1d(in_channels=n_ch[0], out_channels=n_ch[0], kernel_size=15, padding=7, stride=1), activation_layer()))
        # SegFC-LReLU / SegFC-LReLU / SegFC-LReLU-Do-SegFC
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=1, stride=1, padding=0), activation_layer())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=1, stride=1, padding=0), activation_layer())
        self.conv4 = nn.Sequential(nn.Conv1d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=1, padding=0, stride=1), activation_layer(),
            nn.Dropout(), nn.Conv1d(in_channels=n_ch[3], out_channels=n_ch[4], kernel_size=1, padding=0, stride=1))

        self.flatten = nn.Flatten(start_dim=1)

        # Flatten dim - Freq * Channel
        pre_fc_dim = n_bins_in * n_ch[4]

        # pre_fc - Not used (just Identity)
        layers = []
        for i in range(num_output_layers-1):
            layers.extend([
                ToeplitzLinear(pre_fc_dim, pre_fc_dim),
                activation_layer()
            ])
        self.pre_fc = nn.Sequential(*layers)

        # ToeplitzLinear/Softmax
        self.fc = ToeplitzLinear(pre_fc_dim, output_dim)
        self.final_norm = nn.Softmax(dim=-1)

        self.register_buffer("abs_shift", torch.zeros((), dtype=torch.long), persistent=True)

    def forward(self, x):
        """Frame-wise pitch prediction.

        Args:
            x :: (B, 1, Freq=i) - CQT
        Returns:
              :: (B, Freq=o)    - Pitch probability distribution
        """

        # LN-Conv-LReLU-Res[Conv-LReLU]-[SegFC-LReLU]x3-Do-SegFC :: (B, 1, Freq=i) -> (B, Feat=c, Freq=i)
        x = self.conv1(self.layernorm(x))
        for p in range(0, self.n_prefilt_layers - 1):
            x = x + self.prefilt_list[p](x)
        y_pred = self.conv4(self.conv3(self.conv2(x)))

        # Flatten-TeplitzLinear-Softmax :: (B, Feat=c, Freq=i) -> (B, Freq=c*i) -> (B, Freq=o) -> (B, Freq=o)
        dist = self.final_norm(self.fc(self.pre_fc(self.flatten(y_pred))))

        return dist
