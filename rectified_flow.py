import torch 
from torch.distributions import LogisticNormal
class RectifiedFlow(torch.nn.Module):
    def __init__(self, model, ln=True, num_timesteps=1000, t_type = 'U', loc=0.0, scale=1.0):
        super().__init__()
        self.model = model
        self.ln = ln
        self.stratified = False 
        self.num_timesteps = num_timesteps
        self.t_type = t_type
        self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
        self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

    def forward(self, x, pan, ms):

        b = x.size(0)
        if self.t_type == 'U':
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).float() / self.num_timesteps
        elif self.t_type == 'LG':
            t = self.sample_t(x)

        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        vtheta = self.model(zt, t, pan, ms)
        with torch.no_grad():
            cutoff = 0.25 # [B]
            t4 = t.view(b,1,1,1)
            percentile = (30.0 + 70.0 * ((1.0 - t4) / 0.7)).clamp(30.0, 100.0)
            mask = highfreq_mask_2d(x, cutoff_frac=cutoff, percentile=percentile)  # [B,1,H,W]
        #batchwise_mse = (mask *(z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        err = (z1 - x - vtheta)  # [B,C,H,W]
        num_eff = mask.sum(dim=(1,2,3)).clamp_min(1.0)          # [B]
        mse_sum = (mask * err.pow(2)).sum(dim=(1,2,3))          # [B]
        batchwise_mse = mse_sum / num_eff                       # [B]
        return batchwise_mse.mean()

    @torch.no_grad()
    def sample(self, pan, lrms, sample_steps=10):
        z = torch.randn_like(lrms)
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            vc = self.model(z, t, pan, lrms) 
            # if null_cond is not None:
            #     vu = self.model(z, t, null_cond) 
            #     vc = vu + cfg * (vc - vu)
            z = z - dt * vc
            images.append(z)
        return images
    
    @torch.no_grad()
    def sample_with_cfg(self, pan, lrms, cfg = 1.5, sample_steps=10):
        z = torch.randn_like(lrms)
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            vc = self.model.forward_with_cfg(z, t, pan, lrms, cfg) 
            # if null_cond is not None:
            #     vu = self.model(z, t, null_cond) 
            #     vc = vu + cfg * (vc - vu)
            z = z - dt * vc
            images.append(z)
        return images


import torch
import torch.nn.functional as F

def highfreq_mask_2d(x: torch.Tensor, cutoff_frac, percentile: float = 90.0) -> torch.Tensor:
    """
    x: [B,C,H,W]，实数
    cutoff_frac: float 或 Tensor[B]/[B,1]/[B,1,1,1]，越大=越靠外高频，但面积更小
    return: 0/1 掩膜，[B,1,H,W]
    """
    assert x.dim() == 4, f"expect [B,C,H,W], got {x.shape}"
    B, C, H, W = x.shape
    dev, dt = x.device, x.dtype

    # 用 float32 做 FFT 更稳
    xf = x.to(torch.float32)

    X = torch.fft.fft2(xf, dim=(-2, -1), norm='ortho')
    X = torch.fft.fftshift(X, dim=(-2, -1))

    yy = torch.linspace(-1.0, 1.0, H, device=dev, dtype=xf.dtype)
    xx = torch.linspace(-1.0, 1.0, W, device=dev, dtype=xf.dtype)
    Y, Xg = torch.meshgrid(yy, xx, indexing='ij')
    R = torch.sqrt(Xg**2 + Y**2)                     # [H,W]
    Rmax = R.max()

    if isinstance(cutoff_frac, (float, int)):
        r0 = torch.full((B, 1, 1, 1), float(cutoff_frac)*Rmax, device=dev, dtype=xf.dtype)
    else:
        cf = torch.as_tensor(cutoff_frac, device=dev, dtype=xf.dtype)
        cf = cf.view(B, *([1] * (4 - cf.dim())))     # -> [B,1,1,1]
        r0 = cf * Rmax

    hp = (R.view(1, 1, H, W) >= r0).to(xf.dtype)     # [B,1,H,W]
    X_hp = X * hp                                    # broadcast 到 [B,C,H,W]

    X_hp = torch.fft.ifftshift(X_hp, dim=(-2, -1))
    x_hf = torch.fft.ifft2(X_hp, dim=(-2, -1), norm='ortho').abs()  # [B,C,H,W]

    hf_max = x_hf.max(dim=1, keepdim=True).values    # [B,1,H,W]
    flat = hf_max.view(B, -1)                        # [B, N]

    # percentile -> [B] in [0,100]
    if isinstance(percentile, (float, int)):
        q = torch.full((B,), float(percentile), device=dev, dtype=flat.dtype)
    else:
        q = torch.as_tensor(percentile, device=dev, dtype=flat.dtype)
        q = q.view(B, -1).squeeze(-1)                # [B]
    q = q.clamp(0.0, 100.0)

    flat_sorted, _ = torch.sort(flat, dim=1)         # [B, N] 升序
    N = flat_sorted.size(1)
    ki = torch.round((q/100.0) * (N - 1)).long().clamp(0, N - 1)  # [B]
    thr = flat_sorted[torch.arange(B, device=dev), ki].view(B,1,1,1)

    mask = (hf_max <= thr).to(dt)                    # [B,1,H,W]
    return mask