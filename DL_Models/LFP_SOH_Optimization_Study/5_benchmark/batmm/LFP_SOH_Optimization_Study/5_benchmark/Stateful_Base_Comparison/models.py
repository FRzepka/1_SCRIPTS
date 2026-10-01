from __future__ import annotations

import typing as T

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        return output[:, :, : -self.pad] if self.pad > 0 else output


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dropout: float,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.drop1(self.relu1(self.conv1(x)))
        output = self.drop2(self.relu2(self.conv2(output)))
        residual = x if self.downsample is None else self.downsample(x)
        return output + residual


class SOHCNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        mlp_hidden: int,
        kernel_size: int = 5,
        dilations: T.Optional[T.List[int]] = None,
        num_blocks: int = 4,
        dropout: float = 0.15,
        output_kernel_size: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_features, hidden_size, kernel_size=1)
        if dilations is None:
            dilations = [1] * max(1, int(num_blocks))
        self.dilations = [int(value) for value in dilations]
        self.blocks = nn.Sequential(
            *[
                ConvBlock(hidden_size, hidden_size, kernel_size, dropout, dilation)
                for dilation in self.dilations
            ]
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_size, mlp_hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(mlp_hidden, 1, kernel_size=1),
        )
        self.output_kernel_size = int(output_kernel_size)
        self.output_smoother = (
            CausalConv1d(1, 1, self.output_kernel_size)
            if self.output_kernel_size > 1
            else None
        )
        self.kernel_size = int(kernel_size)

    @property
    def receptive_field(self) -> int:
        return (
            1
            + 2 * (self.kernel_size - 1) * sum(self.dilations)
            + self.output_kernel_size
            - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.blocks(self.input_proj(x.transpose(1, 2)))
        output = self.head(output)
        if self.output_smoother is not None:
            output = self.output_smoother(output)
        return output.squeeze(1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.drop(self.fc2(self.act(self.fc1(x)))))


class SOHGRU(nn.Module):
    def __init__(
        self,
        in_features: int,
        embed_size: int,
        hidden_size: int,
        mlp_hidden: int,
        num_layers: int = 2,
        res_blocks: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.post_norm = nn.LayerNorm(output_size)
        self.res_blocks = nn.ModuleList(
            [
                ResidualMLPBlock(output_size, mlp_hidden, dropout)
                for _ in range(max(0, int(res_blocks)))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(output_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: T.Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        output, new_state = self.gru(self.feature_proj(x), state)
        output = self.post_norm(output)
        for block in self.res_blocks:
            output = block(output)
        prediction = self.head(output).squeeze(-1)
        return (prediction, new_state) if return_state else prediction


class SOHLSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        embed_size: int,
        hidden_size: int,
        mlp_hidden: int,
        num_layers: int = 2,
        res_blocks: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.post_norm = nn.LayerNorm(output_size)
        self.res_blocks = nn.ModuleList(
            [
                ResidualMLPBlock(output_size, mlp_hidden, dropout)
                for _ in range(max(0, int(res_blocks)))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(output_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: T.Optional[T.Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        output, new_state = self.lstm(self.feature_proj(x), state)
        output = self.post_norm(output)
        for block in self.res_blocks:
            output = block(output)
        prediction = self.head(output).squeeze(-1)
        return (prediction, new_state) if return_state else prediction


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.dropout1(self.relu1(self.conv1(x)))
        output = self.dropout2(self.relu2(self.conv2(output)))
        residual = x if self.downsample is None else self.downsample(x)
        return output + residual


class SOHTCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        mlp_hidden: int,
        kernel_size: int,
        num_layers: int,
        dilations: T.List[int],
        dropout: float,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilations = [int(value) for value in dilations]
        layers = []
        input_channels = in_features
        for dilation in self.dilations:
            layers.append(
                TemporalBlock(
                    input_channels, hidden_size, kernel_size, dilation, dropout
                )
            )
            input_channels = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(hidden_size, mlp_hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(mlp_hidden, 1, kernel_size=1),
        )

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tcn(x.transpose(1, 2))).squeeze(1)


MODEL_CLASSES = {
    "cnn": SOHCNN,
    "gru": SOHGRU,
    "lstm": SOHLSTM,
    "tcn": SOHTCN,
}
