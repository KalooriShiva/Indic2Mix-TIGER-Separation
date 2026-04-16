import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import ConvNormAct, GlobalLayerNorm, ChannelwiseLayerNorm

class ConvTasNet(nn.Module):
    """
    Conv-TasNet architecture.
    """
    def __init__(
        self,
        N=512,
        L=16,
        B=128,
        H=256,
        P=3,
        X=8,
        R=3,
        C=2,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        sample_rate=16000, # Added for compatibility, not used in architecture
    ):
        super().__init__()
        # Hyper-parameters
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        
        # Components
        self.encoder = Encoder(N, L)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        self.decoder = Decoder(N, L)
        
    def forward(self, mixture):
        """
        Args:
            mixture (torch.Tensor): [B, 1, T], B is batch size, T is #samples
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        return est_source

class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer."""
    def __init__(self, N, L):
        super().__init__()
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture (torch.Tensor): [B, 1, T], B is batch size, T is #samples
        """
        mixture = torch.squeeze(mixture, 1)
        mixture_w = F.relu(self.conv1d_U(mixture.unsqueeze(1)))  # [B, N, L]
        return mixture_w

class Decoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w (torch.Tensor): [B, N, K]
            est_mask (torch.Tensor): [B, C, N, K]
        """
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [B, C, N, K]
        source_w = torch.transpose(source_w, 2, 3) # [B, C, K, N]
        
        # S = DV
        est_source = self.basis_signals(source_w)  # [B, C, K, L]
        est_source = F.fold(
            est_source.contiguous().view(est_source.shape[0] * est_source.shape[1], est_source.shape[2], -1).transpose(1,2),
            (est_source.shape[2]*est_source.shape[3], 1),
            (est_source.shape[3], est_source.shape[3]//2)
        )
        est_source = est_source.view(est_source.shape[0], -1, est_source.shape[2], est_source.shape[3])
        return est_source.squeeze(2).squeeze(1)

class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear='relu'):
        super().__init__()
        self.receptive_field = 0
        self.conv_blocks = nn.ModuleList()
        for r in range(R):
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                self.conv_blocks.append(
                    Conv1DBlock(B, H, P, stride=1, padding=padding, dilation=dilation, norm_type=norm_type, causal=causal)
                )
        
        self.prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(B, C * N, 1)
        
        if mask_nonlinear == 'relu':
            self.nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.nonlinear = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported nonlinear function: {mask_nonlinear}")
            
        self.C = C
        self.N = N

    def forward(self, mixture_w):
        B, N, K = mixture_w.size()
        
        # Pass through LayerNorm and a 1x1 conv
        score = self.prelu(mixture_w)
        
        # Pass through TCN blocks
        for i in range(len(self.conv_blocks)):
            score = self.conv_blocks[i](score)
            
        score = self.mask_conv(score)
        score = score.view(B, self.C, self.N, K)
        est_mask = self.nonlinear(score)
        return est_mask

class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, kernel_size, stride, padding, dilation, norm_type="gLN", causal=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_chan, hid_chan, 1)
        self.prelu1 = nn.PReLU()
        if norm_type == "gLN":
            self.norm1 = GlobalLayerNorm(hid_chan)
        elif norm_type == "cLN":
            self.norm1 = ChannelwiseLayerNorm(hid_chan)
        else: # "BN"
            self.norm1 = nn.BatchNorm1d(hid_chan)
            
        self.dconv = nn.Conv1d(hid_chan, hid_chan, kernel_size, stride, padding, dilation, groups=hid_chan)
        self.prelu2 = nn.PReLU()
        if norm_type == "gLN":
            self.norm2 = GlobalLayerNorm(hid_chan)
        elif norm_type == "cLN":
            self.norm2 = ChannelwiseLayerNorm(hid_chan)
        else: # "BN"
            self.norm2 = nn.BatchNorm1d(hid_chan)
            
        self.conv2 = nn.Conv1d(hid_chan, in_chan, 1)
        self.causal = causal
        self.padding = padding

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.norm1(out)
        out = self.dconv(out)
        out = self.prelu2(out)
        out = self.norm2(out)
        out = self.conv2(out)
        return out + residual