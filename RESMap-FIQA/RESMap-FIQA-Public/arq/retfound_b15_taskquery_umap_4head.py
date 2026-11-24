from __future__ import annotations

import math

import torch
import torch.nn as nn

from arq.retfound_b15_taskquery_4head import (
    RETFoundB15TaskQuery4Head,
    TASK_NAMES,
)


# ============================================================
# Lightweight U-map spatial encoder
# ============================================================

class ConvGNAct(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                out_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class USpatialEncoder(nn.Module):
    """
    1 x 384 x 384
        ->
    32 x 192 x 192
        ->
    64 x 96 x 96
        ->
    128 x 48 x 48
        ->
    256 x 24 x 24
        ->
    576 x 256
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvGNAct(1, 32),
            ConvGNAct(32, 64),
            ConvGNAct(64, 128),
            ConvGNAct(128, 256),
        )

    def forward(self, u):

        x = self.encoder(u)

        if x.shape[-2:] != (24, 24):
            raise RuntimeError(
                "Expected U feature 24x24, "
                f"got {tuple(x.shape)}"
            )

        # [B,256,24,24]
        # -> [B,576,256]
        x = (
            x.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        return x


# ============================================================
# A3
# ============================================================

class RETFoundB15TaskQueryUMap4Head(
    RETFoundB15TaskQuery4Head
):
    """
    A3 = A2 + U-map spatial degradation evidence.

    A2 part is inherited unchanged:

        Frozen RETFound
          -> B15 tokens
          -> four task queries
          -> task-specific spatial pooling
          -> frozen fc_norm
          -> four linear heads

    Added only:

        U-map
          -> U spatial encoder
          -> RGB-U cross-attention
          -> gated residual fusion
    """

    U_DIM = 256
    RGB_DIM = 1024

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 384,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
        )

        # ----------------------------------------------------
        # U-map encoder
        # ----------------------------------------------------
        self.u_encoder = USpatialEncoder()

        # ----------------------------------------------------
        # RGB semantic tokens query U degradation evidence.
        #
        # RGB:
        #   [B,576,1024] -> Q [B,576,256]
        #
        # U:
        #   [B,576,256] -> K/V [B,576,256]
        # ----------------------------------------------------
        self.rgb_q_proj = nn.Linear(
            self.RGB_DIM,
            self.U_DIM,
            bias=True,
        )

        self.u_k_proj = nn.Linear(
            self.U_DIM,
            self.U_DIM,
            bias=True,
        )

        self.u_v_proj = nn.Linear(
            self.U_DIM,
            self.U_DIM,
            bias=True,
        )

        # Map U evidence back to the original
        # 1024-D B15 token space.
        self.u_out_proj = nn.Linear(
            self.U_DIM,
            self.RGB_DIM,
            bias=True,
        )

        # ----------------------------------------------------
        # Spatial scalar gate.
        #
        # Input:
        #   semantic query Q + retrieved U evidence D
        #   [256 + 256]
        #
        # Output:
        #   one gate for every RGB patch.
        # ----------------------------------------------------
        self.gate = nn.Sequential(
            nn.Linear(
                self.U_DIM * 2,
                self.U_DIM,
            ),
            nn.GELU(),
            nn.Linear(
                self.U_DIM,
                1,
            ),
            nn.Sigmoid(),
        )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        # Parent already keeps backbone eval.
        self.backbone.eval()

        return self

    def encode_and_fuse(
        self,
        rgb_tokens,
        u_map,
    ):
        """
        rgb_tokens:
            [B,576,1024]

        u_map:
            [B,1,384,384]
            float32
            expected range approximately [0,1]
        """

        if u_map.ndim != 4:
            raise RuntimeError(
                "Expected U-map [B,1,H,W], "
                f"got {tuple(u_map.shape)}"
            )

        if u_map.shape[1] != 1:
            raise RuntimeError(
                "U-map must have exactly one channel"
            )

        if u_map.shape[-2:] != (384, 384):
            raise RuntimeError(
                "Expected preprocessed U-map 384x384, "
                f"got {tuple(u_map.shape)}"
            )

        if not torch.isfinite(u_map).all():
            raise RuntimeError(
                "Non-finite U-map detected"
            )

        # ----------------------------------------------------
        # U spatial tokens
        # ----------------------------------------------------
        u_tokens = self.u_encoder(
            u_map
        )

        # ----------------------------------------------------
        # Cross-attention
        # ----------------------------------------------------
        q = self.rgb_q_proj(
            rgb_tokens
        )

        k = self.u_k_proj(
            u_tokens
        )

        v = self.u_v_proj(
            u_tokens
        )

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(
            float(self.U_DIM)
        )

        rgb_u_attention = torch.softmax(
            scores,
            dim=-1,
        )

        # [B,576,256]
        u_evidence = torch.matmul(
            rgb_u_attention,
            v,
        )

        # ----------------------------------------------------
        # Gated residual injection
        # ----------------------------------------------------
        gate = self.gate(
            torch.cat(
                [
                    q,
                    u_evidence,
                ],
                dim=-1,
            )
        )

        # [B,576,1024]
        u_residual = self.u_out_proj(
            u_evidence
        )

        fused_tokens = (
            rgb_tokens
            +
            gate * u_residual
        )

        return (
            fused_tokens,
            u_tokens,
            rgb_u_attention,
            gate,
            u_evidence,
        )

    def forward(
        self,
        img,
        u_map,
        *,
        return_attention=False,
        return_features=False,
        return_fusion=False,
    ):

        # ----------------------------------------------------
        # Exact same frozen B15 extraction as A2.
        # ----------------------------------------------------
        rgb_tokens = (
            self.forward_spatial_tokens(
                img
            )
        )

        (
            fused_tokens,
            u_tokens,
            rgb_u_attention,
            gate,
            u_evidence,
        ) = self.encode_and_fuse(
            rgb_tokens,
            u_map,
        )

        # ----------------------------------------------------
        # EXACT A2 task-query decoder.
        # Only input tokens changed:
        #
        # A2: rgb_tokens
        # A3: fused_tokens
        # ----------------------------------------------------
        scores = torch.einsum(
            "td,bnd->btn",
            self.task_queries,
            fused_tokens,
        )

        scores = scores / math.sqrt(
            float(self.EMBED_DIM)
        )

        task_attention = torch.softmax(
            scores,
            dim=-1,
        )

        task_features = torch.einsum(
            "btn,bnd->btd",
            task_attention,
            fused_tokens,
        )

        task_features = (
            self.backbone.fc_norm(
                task_features
            )
        )

        logits = []

        for idx, task in enumerate(
            TASK_NAMES
        ):
            logits.append(
                self.heads[task](
                    task_features[
                        :,
                        idx,
                        :,
                    ]
                )
            )

        logits = torch.cat(
            logits,
            dim=1,
        )

        pred = torch.sigmoid(
            logits
        )

        if not (
            return_attention
            or return_features
            or return_fusion
        ):
            return pred

        out = {
            "pred": pred,
            "logits": logits,
        }

        if return_attention:
            out["task_attention"] = (
                task_attention
            )

            out["task_attention_2d"] = (
                task_attention.reshape(
                    img.shape[0],
                    4,
                    24,
                    24,
                )
            )

        if return_features:
            out["rgb_tokens"] = (
                rgb_tokens
            )

            out["fused_tokens"] = (
                fused_tokens
            )

            out["task_features"] = (
                task_features
            )

            out["u_tokens"] = (
                u_tokens
            )

        if return_fusion:
            out["rgb_u_attention"] = (
                rgb_u_attention
            )

            out["u_gate"] = gate

            out["u_gate_2d"] = (
                gate.transpose(1, 2)
                .reshape(
                    img.shape[0],
                    1,
                    24,
                    24,
                )
            )

            out["u_evidence"] = (
                u_evidence
            )

        return out


def parameter_report(
    model,
):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    backbone_trainable = sum(
        p.numel()
        for p in model.backbone.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "backbone_trainable":
            backbone_trainable,
        "a2_queries":
            model.task_queries.numel(),
        "heads":
            sum(
                p.numel()
                for p in model.heads.parameters()
                if p.requires_grad
            ),
    }
