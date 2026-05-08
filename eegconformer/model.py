# eegconformer/model.py
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=56, emb_size=64):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, emb_size, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            nn.Dropout(0.5),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=(in_channels, 1), groups=emb_size),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = x.squeeze(2)
        return x.permute(0, 2, 1)  # (B, T', emb_size)


class TransformerBlock(nn.Module):
    def __init__(self, dim=64, num_heads=8, ff_dim=256, dropout=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        attn_out, attn_weights = self.attn(
            x, x, x, need_weights=return_attn, average_attn_weights=False
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        if return_attn:
            assert attn_weights is not None  # guard against PyTorch API change
            return x, attn_weights  # attn_weights: (B, num_heads, T', T')
        return x


class EEGConformer(nn.Module):
    """
    EEGConformer: CNN patch embedding + Transformer encoder for EEG classification.
    Input: (batch, in_channels, 260)
    Output: (batch, 1) raw logit  [or (logit, attn_list) when return_attn=True]
    """

    T_PRIME = 13  # floor((260 - 75) / 15) + 1

    def __init__(self, in_channels=56, emb_size=64, num_heads=8,
                 num_layers=4, dropout=0.5):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, emb_size)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.T_PRIME, emb_size))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        self.transformer = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, emb_size * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_size)
        flat_dim = self.T_PRIME * emb_size  # 13 * 64 = 832
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x, return_attn=False):
        x = self.patch_embed(x)   # (B, T', emb_size)
        x = x + self.pos_enc      # learnable positional encoding
        attn_list = []
        for block in self.transformer:
            if return_attn:
                x, attn_w = block(x, return_attn=True)
                attn_list.append(attn_w)  # retains grad_fn; call .detach() for visualization
            else:
                x = block(x)
        x = self.norm(x)
        logit = self.head(x)  # (B, 1)
        if return_attn:
            return logit, attn_list
        return logit
