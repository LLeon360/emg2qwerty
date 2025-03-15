# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder alternative to the TDSConvEncoder

    The expected input shape into the model (after the Rotational Invariant MLP) is (T, N, num_features)
    """

    def __init__(
            self, 
            num_features: int, 
            num_layers: int = 6, 
            d_model: int = 512, 
            nhead: int = 8,
        ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        self.fc_in = nn.Linear(num_features, d_model)
        # Note: Since the CTC loss function expects the input to be of shape (T, N, C), so we set batch_first=False
        encoder_layer = nn.TransformerEncoderLayer(batch_first=False, d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def sinusoidal_positional_encoding(self, d_model: int, length: int) -> torch.Tensor:
        """
        Given d_model and the number of positions (timesteps in signal processing context) returns 1d sinusoidal positional encodings 

        Implementation from "https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py"
        """

        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(inputs) # (T, N, num_features) -> (T, N, d_model)

        # add sinusoidal positional encoding
        pe = self.sinusoidal_positional_encoding(self.d_model, x.shape[0]).unsqueeze(1).to(x.device) # (T, d_model) -> (T, 1, d_model)
        x = x + pe # (T, N, d_model) + (T, 1, d_model) -> (T, N, d_model)
        
        out = self.transformer_encoder(x) # (T, N, d_model) -> (T, N, d_model)

        return out

class RotaryTransformerEncoder(nn.Module):
    """
    Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. ArXiv. https://arxiv.org/abs/2104.09864

    https://github.com/ZhuiyiTechnology/roformer
    """

    """
    Transformer Encoder with Rotary Positional Embeddings
    Input shape: (T, N, num_features) where:
    - T: sequence length (time steps)
    - N: batch size
    - num_features: input feature dimension
    """
    
    def __init__(self, num_features: int, num_layers: int = 6, d_model: int = 512, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Input projection
        self.fc_in = nn.Linear(num_features, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (T, N, num_features)
        Returns:
            Tensor of shape (T, N, d_model)
        """
        # Input projection
        x = self.fc_in(x)  # (T, N, d_model)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)
            
        return x

class RotaryTransformerEncoderLayer(nn.Module):
    """
    Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. ArXiv. https://arxiv.org/abs/2104.09864

    https://github.com/ZhuiyiTechnology/roformer
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = RotaryAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Self-attention branch
        src2 = self.self_attn(self.norm1(src))
        src = src + self.dropout(src2)
        
        # Feedforward branch
        src2 = self.ffn(self.norm2(src))
        src = src + self.dropout(src2)
        return src


class RotaryAttention(nn.Module):
    """
    Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. ArXiv. https://arxiv.org/abs/2104.09864

    https://github.com/ZhuiyiTechnology/roformer
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Projection layers
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Rotary embedding parameters: compute inverse frequencies for half of head_dim.
        self.register_buffer(
            "inv_freq", 
            1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        )

    def _compute_rotary_emb(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Computes rotary frequencies.
        Returns a tensor of shape (seq_len, head_dim) where head_dim is doubled.
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # shape: (seq_len, head_dim/2)
        return torch.cat([freqs, freqs], dim=-1)  # shape: (seq_len, head_dim)

    def _apply_rotary_emb(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary embeddings to tensor x.
        x: Tensor of shape (N, H, T, D)
        freqs: Tensor of shape (T, D)
        Returns: Tensor of shape (N, H, T, D) with rotary embeddings applied.
        """
        # Reshape freqs for broadcasting: (1, 1, T, D)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Splits the last dimension in half and rotates the two halves.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (T, N, E)
            attn_mask: Optional attention mask (default: None)
        Returns:
            Tensor of shape (T, N, E)
        """
        T, N, _ = x.shape
        
        # Project inputs to Q, K, V and split into three tensors
        qkv = self.in_proj(x)  # shape: (T, N, 3*E)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention and rearrange to shape (N, H, T, D)
        q = q.view(T, N, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = k.view(T, N, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = v.view(T, N, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        # Now: q, k, v have shape: (N, H, T, D)
        
        # Compute rotary embeddings for sequence length T and apply to q and k.
        freqs = self._compute_rotary_emb(T, x.device)  # shape: (T, D)
        q = self._apply_rotary_emb(q, freqs)
        k = self._apply_rotary_emb(k, freqs)
        
        # Use PyTorch's efficient scaled_dot_product_attention.
        # It expects inputs of shape (batch, *, seq_len, embed_dim) where * are additional dimensions (here, H)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p)
        # attn_output shape: (N, H, T, D)
        
        # Rearrange output back to (T, N, E)
        attn_output = attn_output.transpose(1, 2).reshape(N, T, self.embed_dim).transpose(0, 1)
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class ConvSubsample(nn.Module):
    """
    2-layer convolutional subsampling in (time, freq) dimension
    Input shape after flatten prolly (T, N, features), but we can reshape it to (N, 1, T, features') for 2D conv
    
    Does not use TDS convolutions, just standard conv layers
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64, stride_time: int = 2, stride_feat: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(stride_time, stride_feat), padding=(1,1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(stride_time, stride_feat), padding=(1,1))
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, features)
        T, N, F = x.shape
        x = x.permute(1, 0, 2)   # (N, T, F)
        x = x.unsqueeze(1)       # (N, 1, T, F)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x: (N, out_channels, T_sub, F_sub)
        N2, C2, T2, F2 = x.shape
        x = x.permute(2, 0, 1, 3)  # (T2, N, C2, F2)
        x = x.reshape(T2, N, C2 * F2)  # (T2, N, C2*F2)
        return x

class CausalTransformerEncoder(nn.Module):
    """
    Transformer encoder that supports causal mask, pre layer norm, and sinusoidal vs learnable vs rotary embeddings
    """
    def __init__(
        self,
        num_features: int,
        num_layers: int = 6,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
        norm_first: bool = False,
        pos_embed: str = "learnable"  # "sinusoidal", "learnable", "rotary"
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.pos_embed = pos_embed.lower()

        if self.pos_embed == "rotary":
            self.encoder = RotaryTransformerEncoder(num_features, num_layers, d_model, nhead)
        else:
            self.fc_in = nn.Linear(num_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                norm_first=norm_first,
                batch_first=False
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # Set up positional embeddings
            self.max_len = 10000
            if self.pos_embed == "learnable":
                self.pos_embedding = nn.Parameter(torch.randn(self.max_len, d_model))
            elif self.pos_embed == "sinusoidal":
                self.pos_embedding = self._sinusoidal_positional_encoding(d_model, self.max_len)
            else:
                raise ValueError("Invalid pos_embed type. Choose from 'sinusoidal', 'learnable', or 'rotary'.")

    def _sinusoidal_positional_encoding(self, d_model: int, length: int) -> torch.Tensor:
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding.")
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe

    def _generate_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, num_features)
        T, N, _ = x.shape
        if self.pos_embed == "rotary":
            # RotaryTransformerEncoder already does its own input projection and processing
            return self.encoder(x)
        else:
            x = self.fc_in(x)  # (T, N, d_model)
            
            # Add positional embedding
            pe = self.pos_embedding[:T, :].to(x.device)  # (T, d_model)
            x = x + pe.unsqueeze(1)
            
            # Apply causal mask if needed
            mask = self._generate_causal_mask(T, x.device) if self.causal else None
            return self.encoder(x, mask=mask)