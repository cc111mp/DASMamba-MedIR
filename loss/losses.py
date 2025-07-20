import torch.nn.functional as F
from torch import nn
import torch
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
from torch.autograd import Variable
from pytorch_wavelets import DWTForward
import random

def create3DsobelFilter():
    num_1, num_2, num_3 = np.zeros((3, 3))
    num_1 = [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
    num_2 = [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]
    num_3 = [[-1., -2., -1.],
             [-2., -4., -2.],
             [-1., -2., -1.]]
    sobelFilter = np.zeros((3, 1, 3, 3, 3))

    sobelFilter[0, 0, 0, :, :] = num_1
    sobelFilter[0, 0, 1, :, :] = num_2
    sobelFilter[0, 0, 2, :, :] = num_3
    sobelFilter[1, 0, :, 0, :] = num_1
    sobelFilter[1, 0, :, 1, :] = num_2
    sobelFilter[1, 0, :, 2, :] = num_3
    sobelFilter[2, 0, :, :, 0] = num_1
    sobelFilter[2, 0, :, :, 1] = num_2
    sobelFilter[2, 0, :, :, 2] = num_3

    return Variable(torch.from_numpy(sobelFilter).type(torch.cuda.FloatTensor))


def sobelLayer(input):
    pad = nn.ConstantPad3d((1, 1, 1, 1, 1, 1), -1)
    kernel = create3DsobelFilter()
    act = nn.Tanh()
    paded = pad(input)
    fake_sobel = F.conv3d(paded, kernel, padding=0, groups=1)/4
    n, c, h, w, l = fake_sobel.size()
    fake = torch.norm(fake_sobel, 2, 1, True)/c*3
    fake_out = act(fake)*2-1
    return fake_out

class EdgeAwareLoss(_WeightedLoss):

    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        self.sobelLayer = sobelLayer
        self.baseloss = nn.L1Loss()

    def forward(self, input, target):
        sobelFake = self.sobelLayer(input)
        sobelReal = self.sobelLayer(target)
        return self.baseloss(sobelFake,sobelReal)

class CharbonnierLoss(nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,input, target):
        value=torch.sqrt(torch.pow(input-target,2)+self.epsilon2)
        return torch.mean(value)


class Centeral_Difference_Loss(nn.Module): 
    def __init__(self):
        super().__init__() 
        

        self.criterion = nn.L1Loss()
        

    def overlap_expand3D(self, x, kernel_size=3, stride=1, padding=1):
        B, C, D, H, W = x.shape 
        num_D=int((D+2*padding-kernel_size)/stride+1) 
        num_H=int((H+2*padding-kernel_size)/stride+1) 
        num_W=int((W+2*padding-kernel_size)/stride+1) 
        
        # import pdb 
        # pdb.set_trace()
        
        x=F.pad(x, (padding, padding, padding, padding, padding, padding))
        x_patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)  ###(position, kernel_size, stride)
        out = x_patches.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous().view(B, C, kernel_size*num_D, kernel_size*num_H, kernel_size*num_W) 
        
        return out 
    
    def forward(self, x, y): 
        
        x_expand = self.overlap_expand3D(x) 
        x_up = F.interpolate(x, scale_factor=3, mode="nearest") 
        x_diff = x_up-x_expand

        y_expand = self.overlap_expand3D(y) 
        y_up = F.interpolate(y, scale_factor=3, mode="nearest") 
        y_diff = y_up-y_expand
        
        return self.criterion(x_diff, y_diff)
    
class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(torch.abs(diff))
        return loss

def sample_with_j(k, n, j):
    if not (0 <= j < k):
        raise ValueError("j must be in the range 0 to k-1.")
    if n > k:
        raise ValueError("n must be less than or equal to k.")
    
    numbers = list(range(k))
    remaining = numbers[:j] + numbers[j+1:]
    sample = [j] + random.sample(remaining, n - 1)
    return sample

class HybridWaveletLoss(nn.Module):
    """Combined Frequency Contrastive and Wavelet Coherence Loss"""
    
    def __init__(self, fcr_weight=1.0, coherence_weight=1.0, wave='db1', mode='zero'):
        super(HybridWaveletLoss, self).__init__()
        self.fcr_weight = fcr_weight
        self.coherence_weight = coherence_weight
        self.l1_loss = nn.L1Loss(reduction='none')
        self.multi_n_num = 2
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        
    def compute_coherence(self, im1, im2):
        # Compute wavelet coefficients
        Yl1, Yh1 = self.dwt(im1)
        Yl2, Yh2 = self.dwt(im2)
        
        # Reshape high-frequency coefficients
        Yh1 = [h.view(h.size(0), -1, h.size(-2), h.size(-1)) for h in Yh1]
        Yh2 = [h.view(h.size(0), -1, h.size(-2), h.size(-1)) for h in Yh2]
        
        # Concatenate low and high-frequency coefficients
        coeffs1 = torch.cat([Yl1] + Yh1, dim=1)
        coeffs2 = torch.cat([Yl2] + Yh2, dim=1)
        
        # Flatten coefficients
        coeffs1_flat = coeffs1.view(coeffs1.size(0), -1)
        coeffs2_flat = coeffs2.view(coeffs2.size(0), -1)
        
        # Compute cross-power spectrum and auto-power spectra
        pxy = torch.mean(torch.abs(coeffs1_flat * coeffs2_flat), dim=1)
        pxx = torch.mean(torch.abs(coeffs1_flat ** 2), dim=1)
        pyy = torch.mean(torch.abs(coeffs2_flat ** 2), dim=1)
        
        # Compute coherence
        coherence = 1 - (pxy) / (torch.sqrt(pxx) * torch.sqrt(pyy))
        return coherence.mean()
    
    def compute_fcr(self, a, p, n):
        a_coeffs = self.dwt(a)[0]
        p_coeffs = self.dwt(p)[0]
        n_coeffs = self.dwt(n)[0]
        
        batch_size = a_coeffs.size(0)
        indices = [sample_with_j(batch_size, self.multi_n_num, i) for i in range(batch_size)]
        indices = torch.tensor(indices).to(a.device).view(-1)
        
        # Reshape for loss computation
        a_coeffs = a_coeffs.view(batch_size, -1)
        p_coeffs = p_coeffs.view(batch_size, -1)
        n_coeffs = n_coeffs.view(batch_size, -1)
        
        d_ap = self.l1_loss(a_coeffs, p_coeffs).mean(dim=1)
        d_an = self.l1_loss(
            a_coeffs.unsqueeze(1).expand(-1, self.multi_n_num, -1),
            n_coeffs[indices].view(batch_size, self.multi_n_num, -1)
        ).mean(dim=2)
        
        return (d_ap.unsqueeze(1) / (d_an + 1e-7)).mean()
    
    def forward(self, anchor, positive, negative):
        fcr_loss = self.compute_fcr(anchor, positive, negative)
        coherence_loss = self.compute_coherence(anchor, positive)
        
        # Combine losses with their respective weights
        total_loss = self.fcr_weight * fcr_loss + self.coherence_weight * coherence_loss
        return total_loss, {'fcr_loss': fcr_loss.item(), 'coherence_loss': coherence_loss.item()}

# -------------------FCR----------------------- #
# Frequency Contrastive Regularization
class FCR(nn.Module):
    def __init__(self, ablation=False):

        super(FCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, a, p, n):
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        contrastive = 0
        for i in range(a_fft.shape[0]):
            d_ap = self.l1(a_fft[i], p_fft[i])
            for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):
                d_an = self.l1(a_fft[i], n_fft[j])
                contrastive += (d_ap / (d_an + 1e-7))
        contrastive = contrastive / (self.multi_n_num * a_fft.shape[0])

        return contrastive
    
######################################################
_reduction_modes = ['none', 'mean', 'sum']

class CoherenceLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CoherenceLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
    def compute_coherence(self, im1, im2):
        pxy = (torch.fft.rfft2(im1) * torch.fft.rfft2(im2).conj() ).abs().mean((1,2,3))
        pxx = (torch.fft.rfft2(im1) * torch.fft.rfft2(im1).conj() ).abs().mean((1,2,3))
        pyy = (torch.fft.rfft2(im2) * torch.fft.rfft2(im2).conj() ).abs().mean((1,2,3))
        return 1 - (pxy) / (pxx**0.5 * pyy**0.5)
    def forward(self, pred, target, **kwargs):
        return self.loss_weight * self.compute_coherence(
            pred, target).mean()


class WaveletCoherenceLoss(nn.Module):
    def __init__(self, wave='db1', J=3, eps=1e-6, low_freq_weight=0.2):
        super().__init__()
        self.dwt = DWTForward(J=J, wave=wave, mode='zero')
        self.J = J
        self.eps = eps
        self.low_freq_weight = low_freq_weight
        self.high_freq_decay = 0.7  # Exponential decay factor for high-freq subbands

    def _safe_coherence(self, sb1, sb2):
        """Numerically stable coherence calculation with feature normalization"""
        # Normalize features
        sb1 = (sb1 - sb1.mean()) / (sb1.std() + self.eps)
        sb2 = (sb2 - sb2.mean()) / (sb2.std() + self.eps)

        # Compute statistics
        pxy = (sb1 * sb2).mean(dim=1)  # (B,)
        pxx = (sb1 ** 2).mean(dim=1)
        pyy = (sb2 ** 2).mean(dim=1)

        # Stabilized coherence calculation
        denominator = torch.clamp(pxx * pyy, min=self.eps**2)
        coherence = (pxy ** 2) / denominator
        return 1 - coherence  # (B,)

    def forward(self, pred, target):
        # Ensure 4D input (B,C,H,W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Wavelet decomposition
        Yl1, Yh1 = self.dwt(pred)
        Yl2, Yh2 = self.dwt(target)

        total_loss = torch.tensor(0.0, device=pred.device)

        # Process low-frequency subband
        low_loss = self._safe_coherence(Yl1.flatten(1), Yl2.flatten(1))
        total_loss += low_loss.mean() * self.low_freq_weight

        # Process high-frequency subbands with exponential decay
        for j in range(len(Yh1)):
            for o in range(Yh1[j].size(2)):  # Orientations
                sb1 = Yh1[j][:, :, o, :, :].flatten(1)
                sb2 = Yh2[j][:, :, o, :, :].flatten(1)
                
                subband_loss = self._safe_coherence(sb1, sb2)
                weight = self.high_freq_decay ** j  # Decay by decomposition level
                total_loss += subband_loss.mean() * weight

        return total_loss / (len(Yh1) + 1)  # Normalize by total subband groups