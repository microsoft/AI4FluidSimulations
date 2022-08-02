import numpy as np
import cupy as cp
import torch, distdl, math, time
from mpi4py import MPI
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.repartition_cupy import Repartition
from distdl.nn.broadcast import Broadcast
from distdl.utilities.torch import zero_volume_tensor
from distdl.utilities.slicing import *
import torch.nn.functional as F
from distdl.functional import ZeroVolumeCorrectorFunction


class DistributedRelativeLpLoss(distdl.nn.Module):
        
    def __init__(self, P_root, P_x, p=2):
        super(DistributedRelativeLpLoss, self).__init__()
        
        self.P_0 = P_root
        self.P_x = P_x
        self.p = p
        
        self.sr0 = distdl.nn.SumReduce(P_x, self.P_0)
        self.sr1 = distdl.nn.SumReduce(P_x, self.P_0)
    
    def forward(self, y_hat, y):
        batch_size = y_hat.shape[0]
        y_hat_flat = y_hat.reshape(batch_size, -1)
        y_flat = y.reshape(batch_size, -1)
        
        num = torch.sum(torch.pow(torch.abs(y_hat_flat-y_flat), self.p), dim=1)
        denom = torch.sum(torch.pow(torch.abs(y_flat), self.p), dim=1)
        num_global = self.sr0(num)
        denom_global = self.sr1(denom)
            
        if self.P_0.active:
            num_global = torch.pow(num_global, 1/self.p)
            denom_global = torch.pow(denom_global, 1/self.p)
            
        out = torch.mean(num_global/denom_global)
        return ZeroVolumeCorrectorFunction.apply(out)


class Linear(distdl.nn.Module):

    def __init__(self, P_root, P_feat, channel_in, channel_out, broadcast_weights=None, broadcast_bias=None):
        super(Linear, self).__init__()
        device = torch.device(f'cuda:{P_feat.rank}')

        if P_root.active:
            self.w = torch.nn.Parameter(torch.empty(channel_in, channel_out, 1, 1, 1, 1, device=device))
            self.b = torch.nn.Parameter(torch.zeros(1, channel_out, 1, 1, 1, 1, device=device))
            torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        else:
            self.register_buffer('w', torch.nn.Parameter(zero_volume_tensor(device=device)))
            self.register_buffer('b', torch.nn.Parameter(zero_volume_tensor(device=device)))

        if broadcast_weights is None and broadcast_bias is None:
            self.broadcast_weights = Broadcast(P_root, P_feat)
            self.broadcast_bias = Broadcast(P_root, P_feat)
        else:
            self.broadcast_weights = broadcast_weights
            self.broadcast_bias = broadcast_bias

    def forward(self, x):
        # Assumes that x is already partitioned across P_feat
        # Weights are stored on the root partition only

        # Broadcast weights to all workers
        w = self.broadcast_weights(self.w)
        b = self.broadcast_bias(self.b)

        # Linear encoding
        x = torch.einsum("bixyzt,ioxyzt->boxyzt", x, w) + b
        return x


class SpectralConv(distdl.nn.Module):

    def __init__(self, P_spec, transposef, transposeb, channel_hidden, num_k):
        super(SpectralConv, self).__init__()

        self.device = torch.device(f'cuda:{P_spec.rank}')

        # Repartitioning
        self.transposef = transposef    # Repartition(P_in, P_spec)
        self.transposeb = transposeb    # Repartition(P_spec, P_in)

        # FFTs
        self.fft = torch.fft.fftn
        self.rfft = torch.fft.rfftn
        self.ifft = torch.fft.ifftn
        self.irfft = torch.fft.irfftn
        
        # Compute weight size per worker
        xstart_index = compute_start_index(P_spec.shape[2], P_spec.index[2], 2*num_k[0])[0]
        xstop_index = compute_stop_index(P_spec.shape[2], P_spec.index[2], 2*num_k[0])[0]
        self.x_shape_local = xstop_index - xstart_index

        # Remaining modes
        self.num_kx = num_k[0]
        self.num_ky = num_k[1]
        self.num_kz = num_k[2]
        self.num_kw = num_k[3]

        # Initialize modes
        scaler = 1.0 / channel_hidden / channel_hidden
        self.w1 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.x_shape_local, num_k[1], 2*num_k[2], num_k[3], 
            device=self.device, dtype=torch.complex64))
        self.w2 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.x_shape_local, num_k[1], 2*num_k[2], num_k[3],
            device=self.device, dtype=torch.complex64))


    def forward(self, x):

        # FFT along all but partitioned dimension
        x = self.rfft(x, dim=(2,4,5))
        
        # Subsample modes along all but partitioned dimensions
        xsub = torch.zeros(x.shape[0], x.shape[1], 2*self.num_kx, x.shape[3], 2*self.num_kz, self.num_kw, device=x.device, dtype=torch.complex64)
        xsub[:, :, :self.num_kx, :, :self.num_kz, :] = x[:, :, :self.num_kx, :, :self.num_kz, :self.num_kw]
        xsub[:, :, :self.num_kx, :, -self.num_kz:, :] = x[:, :, :self.num_kx, :, -self.num_kz:, :self.num_kw]
        xsub[:, :, -self.num_kx:, :, :self.num_kz, :] = x[:, :, -self.num_kx:, :, :self.num_kz, :self.num_kw]
        xsub[:, :, -self.num_kx:, :, -self.num_kz:, :] = x[:, :, -self.num_kx:, :, -self.num_kz:, :self.num_kw]

        # Repartition and FFT along remaining dimension
        xsub = self.transposef(xsub)
        xsub = self.fft(xsub, dim=(3))

        # Spectral convolution
        ysub = torch.clone(xsub)*0
        ysub[:, :, :, :self.num_ky, :, :] = torch.einsum("bixyzt,ioxyzt->boxyzt", xsub[:, :, :, :self.num_ky, :, :], self.w1)
        ysub[:, :, :, -self.num_ky:, :, :] = torch.einsum("bixyzt,ioxyzt->boxyzt", xsub[:, :, :, -self.num_ky:, :, :], self.w2)

        # iFFT along non-subsampled dimension
        ysub = self.ifft(ysub, dim=(3))
        ysub = self.transposeb(ysub)

        # Expand subsampled dimensions
        y = torch.zeros(x.shape, device=x.device, dtype=torch.complex64)
        y[:, :, :self.num_kx, :, :self.num_kz, :self.num_kw] = ysub[:, :, :self.num_kx, :, :self.num_kz, :]
        y[:, :, :self.num_kx, :, -self.num_kz:, :self.num_kw] = ysub[:, :, :self.num_kx, :, -self.num_kz:, :]
        y[:, :, -self.num_kx:, :, :self.num_kz, :self.num_kw] = ysub[:, :, -self.num_kx:, :, :self.num_kz, :]
        y[:, :, -self.num_kx:, :, -self.num_kz:, :self.num_kw] = ysub[:, :, -self.num_kx:, :, -self.num_kz:, :]

        # iFFT along remaining dimensions
        y = self.irfft(y, dim=(2,4,5))
        return y


class FNOLayer4D(distdl.nn.Module):

    def __init__(self, P_root, P_x, P_y, transposef, transposeb, broadcast_weights, broadcast_bias, channel_hidden, num_k):
        super(FNOLayer4D, self).__init__()

        self.spectral_conv = SpectralConv(P_y, transposef, transposeb, channel_hidden, num_k)
        self.linear = Linear(P_root, P_x, channel_hidden, channel_hidden, broadcast_weights=broadcast_weights, broadcast_bias=broadcast_bias)

    def forward(self, x):

        xb = self.linear(x)
        x = self.spectral_conv(x)
        x = F.relu(x + xb)

        return x


class ParallelFNO4d(distdl.nn.Module):

    def __init__(self, P_world, P_root, P_x, P_y, channel_in, channel_hidden, channel_out, num_k, init_weights=True, padding=6, balance=False):
        super(ParallelFNO4d, self).__init__()
        P_world._comm.Barrier()

        # Reuse distributors for FNO layers to avoid creating expensive nccl communicators multiple times
        transposef = Repartition(P_x, P_y)
        transposeb = Repartition(P_y, P_x)
        broadcast_weights = Broadcast(P_root, P_x)
        broadcast_bias = Broadcast(P_root, P_x)

        # Encoder
        self.encoder1 = Linear(P_root, P_x, channel_in, channel_hidden // 2)
        self.encoder2 = Linear(P_root, P_x, channel_hidden // 2, channel_hidden)
        self.fno1 = FNOLayer4D(P_root, P_x, P_y, transposef, transposeb, broadcast_weights, broadcast_bias, channel_hidden, num_k)
        self.fno2 = FNOLayer4D(P_root, P_x, P_y, transposef, transposeb, broadcast_weights, broadcast_bias, channel_hidden, num_k)
        self.fno3 = FNOLayer4D(P_root, P_x, P_y, transposef, transposeb, broadcast_weights, broadcast_bias, channel_hidden, num_k)
        self.fno4 = FNOLayer4D(P_root, P_x, P_y, transposef, transposeb, broadcast_weights, broadcast_bias, channel_hidden, num_k)
        self.decoder1 = Linear(P_root, P_x, channel_hidden, 32)
        self.decoder2 = Linear(P_root, P_x, 32, channel_out)
        self.padding = padding

    def forward(self, x):
        
        # Encoder
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))

        # Padding along time
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])

        # FNO Layers
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.fno3(x)
        x = self.fno4(x)
        
        # Remove padding along time
        if self.padding > 0:
            x = x[..., :-self.padding]

        # Decoder
        x = F.relu(self.decoder1(x))
        x = self.decoder2(x)

        return x