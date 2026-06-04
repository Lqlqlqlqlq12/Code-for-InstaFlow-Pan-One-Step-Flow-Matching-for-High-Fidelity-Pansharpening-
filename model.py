import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
import numbers
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate_2d(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb          #B*N*Dim

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, dim_in, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim_in, dim*2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x, y):
        b,c,h,w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k,v = kv.chunk(2, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class DiTLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
    
class Refineblock(nn.Module):
    """
        selfattention + crossattention + gdfn
        self norm 
        cross norm q
                   k,v norm it self
        sqrt nn.param
    """
    def __init__(self, dim, dim_cond, num_heads, fffn_expansion_factor=2, self_bias=False, cross_bias=False, ffn_bias = False):
        super().__init__()
        self.norm1 = DiTLayerNorm(dim)
        self.self_attention = Attention(dim=dim,num_heads=num_heads,bias=self_bias)
        self.norm2 = DiTLayerNorm(dim)
        self.ffn = FeedForward(dim=dim,ffn_expansion_factor=fffn_expansion_factor,bias=ffn_bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, 6 * dim, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * self.self_attention(modulate_2d(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.ffn(modulate_2d(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DR.
    """
    def __init__(self, dim_cond,hidden_size, out_channels):
        super().__init__()
        self.norm_final = DiTLayerNorm(hidden_size)
        self.conv2d = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate_2d(self.norm_final(x), shift, scale)
        x = self.conv2d(x)
        return x

class DRblock_pas_share_t(nn.Module):
    """
        selfattention + crossattention + gdfn
        self norm 
        cross norm q
                   k,v norm it self
        sqrt nn.param
    """
    def __init__(self, dim, dim_cross_in,dim_cond, num_heads, fffn_expansion_factor=2, self_bias=False, cross_bias=False, ffn_bias = False):
        super().__init__()
        self.norm1 = DiTLayerNorm(dim)
        self.self_attention = Attention(dim=dim,num_heads=num_heads,bias=self_bias)
        self.norm2 = DiTLayerNorm(dim)
        self.norm_ms_kv = LayerNorm(dim_cross_in, LayerNorm_type="WithBias")
        self.cross_attention_ms = CrossAttention(dim=dim,dim_in=dim_cross_in,num_heads=num_heads,bias=cross_bias)
        self.norm3 = DiTLayerNorm(dim)
        self.norm4 = DiTLayerNorm(dim)
        self.ffn1 = FeedForward(dim=dim,ffn_expansion_factor=fffn_expansion_factor,bias=ffn_bias)
        self.ffn2 = FeedForward(dim=dim,ffn_expansion_factor=fffn_expansion_factor,bias=ffn_bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, 6 * dim, bias=True)
        )

    def forward(self, x, c, ms):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * self.self_attention(modulate_2d(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.ffn1(modulate_2d(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * self.cross_attention_ms(modulate_2d(self.norm3(x), shift_msa, scale_msa),self.norm_ms_kv(ms)) #+ gate_msa.unsqueeze(-1).unsqueeze(-1) *self.cross_attention_pan(modulate_2d(self.norm3(x), shift_msa, scale_msa),self.norm_pan_kv(pan))
        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.ffn2(modulate_2d(self.norm4(x), shift_mlp, scale_mlp))
        return x
from pre_pvt import net_cond
class DRp_single_pan_and_ms_share_t(nn.Module):
    """
    Diffusion model with a Transformer / Restormer backbone.
    """
    def __init__(
        self,
        input_channels=4,
        input_pan=1,
        input_ms=4,
        output_channels=4,
        dim=[32,64,128],
        dim_t=256,
        dim_cond=[0,0,0],
        num_prompt=[256,128,64],
        dim_prompt=[256,256,256],
        num_heads=[4,8,16],
        num_block=[2,2,2],
        num_refine_block=2,
        ffn_expansion_factor=2.0,
        dim_cross= [64,128,320],
    ):
        super().__init__()
        self.in_channels  = input_channels
        self.out_channels = output_channels
        self.gen_cond_pan_ms = net_cond(in_chans = input_pan + input_ms)
        self.x_embedder   = OverlapPatchEmbed(in_c=input_channels,embed_dim=dim[0])
        self.t_embedder   = TimestepEmbedder(dim_t)
        self.encoder_level1 = nn.ModuleList([DRblock_pas_share_t(dim=dim[0], dim_cross_in=dim_cross[0],dim_cond=dim_t+dim_cond[0], num_heads=num_heads[0], fffn_expansion_factor=ffn_expansion_factor) for _ in range(num_block[0])])
        self.down1_2 = Downsample(dim[0])
        
        self.encoder_level2 = nn.ModuleList([DRblock_pas_share_t(dim=dim[1], dim_cross_in=dim_cross[1],dim_cond=dim_t+dim_cond[1], num_heads=num_heads[1], fffn_expansion_factor=ffn_expansion_factor) for _ in range(num_block[1])])
        self.down2_3 = Downsample(dim[1])
        
        self.latent = nn.ModuleList([DRblock_pas_share_t(dim=dim[2], dim_cross_in=dim_cross[2],dim_cond=dim_t+dim_cond[2], num_heads=num_heads[2], fffn_expansion_factor=ffn_expansion_factor) for _ in range(num_block[2])])
        
        self.up3_2 = Upsample(int(dim[2]))
        self.reduce_chan_level2 = nn.Conv2d(int(dim[2]), int(dim[1]), kernel_size=1, bias=False)
        self.decoder_level2 = nn.ModuleList([DRblock_pas_share_t(dim=dim[1], dim_cross_in=dim_cross[1],dim_cond=dim_t+dim_cond[1], num_heads=num_heads[1], fffn_expansion_factor=ffn_expansion_factor) for _ in range(num_block[1])])
        
        self.up2_1 = Upsample(int(dim[1]))
        #self.reduce_chan_level1 = nn.Conv2d(int(dim[1]), int(dim[0]), kernel_size=1, bias=False)
        self.decoder_level1 = nn.ModuleList([DRblock_pas_share_t(dim=dim[0]*2, dim_cross_in=dim_cross[0],dim_cond=dim_t+dim_cond[0], num_heads=num_heads[0], fffn_expansion_factor=ffn_expansion_factor) for _ in range(num_block[0])])
        
        self.refine = nn.ModuleList([Refineblock(dim=dim[0]*2, dim_cond=dim_t+dim_cond[0], num_heads=num_heads[0], fffn_expansion_factor=ffn_expansion_factor) for _ in range(num_refine_block)])
        
        self.output = FinalLayer(hidden_size=dim[0]*2,dim_cond=dim_t+dim_cond[0],out_channels=output_channels)
        self.initialize_weights()
    
    def initialize_weights(self):

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.encoder_level1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for block in self.encoder_level2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.latent:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.decoder_level2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.decoder_level1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for block in self.refine:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

       
        nn.init.constant_(self.output.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output.adaLN_modulation[-1].bias, 0)


    def forward(self, x, t, pan, ms):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        pan: (N, 1, H, W) tensor of cond
        ms: (N, C, H, W) tensor of cond
        """
        batch_size = x.shape[0]
        [pan_level1, pan_level2, pan_level3] = self.gen_cond_pan_ms(torch.cat([pan, ms], dim=1))
        c = self.t_embedder(t)
        out_en_level1 = self.x_embedder(x)
        for block in self.encoder_level1:
            out_en_level1 = block(out_en_level1, c, pan_level1)
        out_en_level2 = self.down1_2(out_en_level1) 
        for block in self.encoder_level2:
            out_en_level2 = block(out_en_level2, c, pan_level2)
        latent = self.down2_3(out_en_level2)
        for block in self.latent:
            latent = block(latent, c, pan_level3)
        out_de_level2 = self.up3_2(latent)
        out_de_level2 = torch.cat([out_de_level2, out_en_level2], dim=1)
        out_de_level2 = self.reduce_chan_level2(out_de_level2)
        for block in self.decoder_level2:
            out_de_level2 = block(out_de_level2, c, pan_level2)
        out_de_level1 = self.up2_1(out_de_level2)
        out_de_level1 = torch.cat([out_de_level1, out_en_level1], dim=1)
        for block in self.decoder_level1:
            out_de_level1 = block(out_de_level1, c, pan_level1)
        
        for block in self.refine:
            out_de_level1 = block(out_de_level1, c)
        out_x = self.output(out_de_level1, c)
        
        return out_x
