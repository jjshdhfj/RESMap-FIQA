from __future__ import annotations

import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from arq.retfound_4head_clean import (
    RETFound4HeadClean,
    TASK_NAMES,
)


class RETFoundB15TaskQuery4Head(nn.Module):
    """
    A2:
        Frozen RETFound
            ->
        B15 spatial patch tokens [B,576,1024]
            ->
        Four learnable task queries
            ->
        Task-specific spatial attention pooling
            ->
        Same frozen fc_norm as A1
            ->
        Four Linear(1024,1)
            ->
        Sigmoid

    Compared with A1:
        A1 = uniform GAP
        A2 = learned task-specific weighted pooling

    No U-map.
    No Response.
    No projection bottleneck.
    No FFN.
    No task-relation module.
    """

    BLOCK = 15
    EMBED_DIM = 1024
    NUM_PATCH_TOKENS = 576
    NUM_TASKS = 4

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 384,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Exact same RETFound construction/loading as A0/A1
        # ----------------------------------------------------
        base = RETFound4HeadClean(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
        )

        self.backbone = base.backbone

        # Do not inherit already-created A0 heads.
        del base

        # ----------------------------------------------------
        # Frozen RETFound
        # ----------------------------------------------------
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

        if len(self.backbone.blocks) != 24:
            raise RuntimeError(
                f"Expected 24 blocks, got "
                f"{len(self.backbone.blocks)}"
            )

        if not hasattr(self.backbone, "fc_norm"):
            raise RuntimeError(
                "RETFound backbone has no fc_norm"
            )

        # ----------------------------------------------------
        # Capture B15 output
        # ----------------------------------------------------
        self._selected_tokens = None

        target_block = self.backbone.blocks[
            self.BLOCK - 1
        ]

        self._hook_handle = (
            target_block.register_forward_hook(
                self._capture_tokens
            )
        )

        # ----------------------------------------------------
        # Four learnable queries.
        #
        # No K/V projection:
        #
        # Q = task query
        # K = original B15 tokens
        # V = original B15 tokens
        #
        # This keeps A2 extremely close to A1.
        # ----------------------------------------------------
        self.task_queries = nn.Parameter(
            torch.empty(
                self.NUM_TASKS,
                self.EMBED_DIM,
            )
        )

        # ----------------------------------------------------
        # Same head form as A0/A1
        # ----------------------------------------------------
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(
                    self.EMBED_DIM,
                    1,
                    bias=True,
                )
                for task in TASK_NAMES
            }
        )

        self.reset_parameters()

        backbone_trainable = sum(
            p.numel()
            for p in self.backbone.parameters()
            if p.requires_grad
        )

        if backbone_trainable != 0:
            raise RuntimeError(
                "RETFound must remain frozen"
            )

    def reset_parameters(self):

        # Small random initialization means the initial
        # attention is close to uniform GAP rather than
        # strongly biased to arbitrary patches.
        trunc_normal_(
            self.task_queries,
            std=0.02,
        )

        # Same head initialization as A0/A1.
        for head in self.heads.values():
            trunc_normal_(
                head.weight,
                std=2e-5,
            )
            nn.init.zeros_(
                head.bias
            )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        # Backbone always stays eval because it is frozen.
        self.backbone.eval()

        return self

    def _capture_tokens(
        self,
        module,
        inputs,
        output,
    ):
        del module, inputs

        if isinstance(output, (tuple, list)):
            if len(output) == 0:
                raise RuntimeError(
                    "Empty B15 output"
                )
            output = output[0]

        if not torch.is_tensor(output):
            raise RuntimeError(
                "B15 output must be tensor"
            )

        self._selected_tokens = output

    def forward_spatial_tokens(
        self,
        img: torch.Tensor,
    ):

        self._selected_tokens = None

        # Run exactly the normal frozen RETFound feature path.
        with torch.no_grad():
            _ = self.backbone.forward_features(
                img
            )

        tokens = self._selected_tokens

        if tokens is None:
            raise RuntimeError(
                "Failed to capture B15 tokens"
            )

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected [B,N,1024], "
                f"got {tuple(tokens.shape)}"
            )

        if tokens.shape[-1] != self.EMBED_DIM:
            raise RuntimeError(
                f"Unexpected embedding dim: "
                f"{tokens.shape[-1]}"
            )

        # CLS + 576 patches
        if (
            tokens.shape[1]
            ==
            self.NUM_PATCH_TOKENS + 1
        ):
            tokens = tokens[:, 1:, :]

        elif (
            tokens.shape[1]
            !=
            self.NUM_PATCH_TOKENS
        ):
            raise RuntimeError(
                f"Unexpected token count: "
                f"{tokens.shape[1]}"
            )

        return tokens

    def forward(
        self,
        img: torch.Tensor,
        *,
        return_attention: bool = False,
        return_features: bool = False,
    ):

        # ----------------------------------------------------
        # B15 spatial tokens
        #
        # [B,576,1024]
        # ----------------------------------------------------
        tokens = self.forward_spatial_tokens(
            img
        )

        B = tokens.shape[0]

        # ----------------------------------------------------
        # Task-query attention
        #
        # Q: [4,1024]
        # K: [B,576,1024]
        #
        # scores:
        # [B,4,576]
        # ----------------------------------------------------
        scores = torch.einsum(
            "td,bnd->btn",
            self.task_queries,
            tokens,
        )

        scores = scores / math.sqrt(
            float(self.EMBED_DIM)
        )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        # ----------------------------------------------------
        # Weighted spatial pooling
        #
        # V = original B15 tokens
        #
        # [B,4,576] × [B,576,1024]
        # ->
        # [B,4,1024]
        # ----------------------------------------------------
        task_features = torch.einsum(
            "btn,bnd->btd",
            attention,
            tokens,
        )

        # IMPORTANT:
        # fc_norm parameters are frozen,
        # but do NOT use torch.no_grad here,
        # otherwise query gradients are cut off.
        task_features = self.backbone.fc_norm(
            task_features
        )

        logits = []

        for idx, task in enumerate(
            TASK_NAMES
        ):
            logits.append(
                self.heads[task](
                    task_features[:, idx, :]
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
        ):
            return pred

        out = {
            "pred": pred,
            "logits": logits,
        }

        if return_features:
            out["tokens"] = tokens
            out["task_features"] = (
                task_features
            )

        if return_attention:
            out["attention"] = attention

            out["attention_2d"] = (
                attention.reshape(
                    B,
                    self.NUM_TASKS,
                    24,
                    24,
                )
            )

        return out


def parameter_report(
    model: nn.Module,
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

    query_trainable = (
        model.task_queries.numel()
    )

    head_trainable = sum(
        p.numel()
        for p in model.heads.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "backbone_trainable":
            backbone_trainable,
        "query_trainable":
            query_trainable,
        "head_trainable":
            head_trainable,
    }
