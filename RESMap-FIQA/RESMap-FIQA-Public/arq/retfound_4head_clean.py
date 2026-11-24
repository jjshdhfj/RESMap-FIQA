from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

import models_vit
from timm.models.layers import trunc_normal_

from util.pos_embed import interpolate_pos_embed


TASK_NAMES = (
    "clarity",
    "exposure",
    "artifact",
    "completeness",
)


def _extract_state_dict(checkpoint):
    """
    Support several common RETFound checkpoint layouts.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be dict, got {type(checkpoint)}"
        )

    # Original RETFound MAE checkpoint.
    if (
        "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):
        return checkpoint["model"]

    # Other common wrappers.
    for key in (
        "state_dict",
        "params",
        "params_ema",
    ):
        if (
            key in checkpoint
            and isinstance(checkpoint[key], dict)
        ):
            return checkpoint[key]

    # Raw state dict.
    if all(
        torch.is_tensor(v)
        for v in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Cannot identify RETFound state_dict "
        f"from checkpoint keys: {list(checkpoint.keys())[:20]}"
    )


def _clean_state_dict(state):
    """
    Remove common wrappers/prefixes.
    """

    cleaned = {}

    for key, value in state.items():

        k = key

        # Distributed prefix.
        if k.startswith("module."):
            k = k[len("module."):]

        # Some RETFound checkpoints may wrap the backbone.
        if k.startswith("backbone."):
            k = k[len("backbone."):]

        # Compatibility with some ViT variants.
        k = k.replace(
            "mlp.w12.",
            "mlp.fc1.",
        )
        k = k.replace(
            "mlp.w3.",
            "mlp.fc2.",
        )

        cleaned[k] = value

    return cleaned


def load_retfound_pretrained(
    backbone: nn.Module,
    checkpoint_path: str,
):
    """
    Load RETFound weights into a global_pool=True ViT-L.

    Important:
    the pretraining checkpoint may contain final `norm.*`
    while global_pool=True uses `fc_norm.*`.

    For the frozen clean baseline we copy compatible
    pretrained norm weights into fc_norm so that fc_norm
    is not a randomly initialized frozen layer.
    """

    checkpoint_path = Path(
        checkpoint_path
    ).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    print(
        f"[RETFound] loading checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
    )

    state = _extract_state_dict(
        checkpoint
    )

    state = _clean_state_dict(
        state
    )

    # We do not use the original classifier.
    state.pop(
        "head.weight",
        None,
    )
    state.pop(
        "head.bias",
        None,
    )

    # --------------------------------------------------------
    # Global-pool compatibility.
    #
    # Original MAE checkpoint commonly has:
    #   norm.weight / norm.bias
    #
    # global_pool=True model has:
    #   fc_norm.weight / fc_norm.bias
    #
    # Copy them when compatible.
    # --------------------------------------------------------

    if (
        hasattr(backbone, "fc_norm")
        and
        "fc_norm.weight" not in state
        and
        "norm.weight" in state
    ):
        if (
            state["norm.weight"].shape
            ==
            backbone.fc_norm.weight.shape
        ):
            state[
                "fc_norm.weight"
            ] = state[
                "norm.weight"
            ].clone()

            print(
                "[RETFound] mapped "
                "norm.weight -> fc_norm.weight"
            )

    if (
        hasattr(backbone, "fc_norm")
        and
        "fc_norm.bias" not in state
        and
        "norm.bias" in state
    ):
        if (
            state["norm.bias"].shape
            ==
            backbone.fc_norm.bias.shape
        ):
            state[
                "fc_norm.bias"
            ] = state[
                "norm.bias"
            ].clone()

            print(
                "[RETFound] mapped "
                "norm.bias -> fc_norm.bias"
            )

    # global_pool=True removes backbone.norm.
    state.pop(
        "norm.weight",
        None,
    )
    state.pop(
        "norm.bias",
        None,
    )

    # RETFound checkpoint may be 224 while current FIQA
    # input is 384. Keep original RETFound interpolation.
    interpolate_pos_embed(
        backbone,
        state,
    )

    result = backbone.load_state_dict(
        state,
        strict=False,
    )

    print(
        "[RETFound] missing keys:",
        result.missing_keys,
    )

    print(
        "[RETFound] unexpected keys:",
        result.unexpected_keys,
    )

    return result


class RETFound4HeadClean(
    nn.Module):
    """
    Minimal four-dimensional FIQA baseline:

        RGB
          ->
        Frozen RETFound ViT-L/16
          ->
        native F24 patch GAP
          ->
        fc_norm
          ->
        1024-D
          ->
        four independent Linear(1024, 1)

    No local branch.
    No F17/F24 routing.
    No Response.
    No U-map.
    No TaskDecoder.
    No TDRGF.
    """

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 384,
    ):
        super().__init__()

        self.image_size = int(
            image_size
        )

        # ----------------------------------------------------
        # Original RETFound downstream representation:
        # global_pool=True
        # ----------------------------------------------------

        self.backbone = (
            models_vit.RETFound_mae(
                img_size=self.image_size,
                num_classes=0,
                drop_path_rate=0.0,
                global_pool=True,
            )
        )

        if not self.backbone.global_pool:
            raise RuntimeError(
                "Clean RETFound baseline must use "
                "global_pool=True."
            )

        load_retfound_pretrained(
            self.backbone,
            checkpoint_path,
        )

        # ----------------------------------------------------
        # Freeze the entire RETFound encoder, including
        # fc_norm.
        # ----------------------------------------------------

        for param in (
            self.backbone.parameters()
        ):
            param.requires_grad = False

        self.backbone.eval()

        feature_dim = int(
            self.backbone.embed_dim
        )

        if feature_dim != 1024:
            raise RuntimeError(
                "Expected RETFound ViT-L feature dim 1024, "
                f"got {feature_dim}"
            )

        # ----------------------------------------------------
        # Four independent scalar regression heads.
        #
        # No 1024 -> 256 bottleneck.
        # ----------------------------------------------------

        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(
                    feature_dim,
                    1,
                    bias=True,
                )
                for task
                in TASK_NAMES
            }
        )

        # Follow original RETFound classifier-head
        # initialization scale.
        for head in (
            self.heads.values()
        ):
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
        """
        Heads may train, but frozen RETFound always remains
        in eval mode.
        """

        super().train(mode)

        self.backbone.eval()

        return self

    def forward_features(
        self,
        img,
    ):
        """
        Native RETFound global-pool representation.

        models_vit.py returns:
            [B, 1, 1024]
        when global_pool=True.
        """

        with torch.no_grad():

            feat = (
                self.backbone
                .forward_features(
                    img
                )
            )

        if feat.ndim == 3:

            if feat.shape[1] != 1:
                raise RuntimeError(
                    "Expected global pooled RETFound "
                    f"feature [B,1,1024], got {tuple(feat.shape)}"
                )

            feat = feat[:, 0, :]

        if (
            feat.ndim != 2
            or
            feat.shape[-1] != 1024
        ):
            raise RuntimeError(
                "Expected RETFound feature [B,1024], "
                f"got {tuple(feat.shape)}"
            )

        return feat

    def forward(
        self,
        img,
        *,
        return_features=False,
    ):

        feat = self.forward_features(
            img
        )

        logits = torch.cat(
            [
                self.heads[task](
                    feat
                )
                for task
                in TASK_NAMES
            ],
            dim=1,
        )

        # Current four-label FIQA pipeline trains normalized
        # 0~1 targets (original labels are 0~5).
        # Sigmoid only establishes the score range; there is
        # still no hidden MLP/projection.
        pred = torch.sigmoid(
            logits
        )

        if return_features:

            return {
                "pred": pred,
                "logits": logits,
                "feature": feat,
            }

        return pred


def parameter_report(
    model,
) -> Dict[str, int]:

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
        for p
        in model.backbone.parameters()
        if p.requires_grad
    )

    head_trainable = sum(
        p.numel()
        for p
        in model.heads.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "backbone_trainable":
            backbone_trainable,
        "head_trainable":
            head_trainable,
    }
