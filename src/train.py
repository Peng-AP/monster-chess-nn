import argparse
import copy
import json
import os
import random
import re
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from config import (
    TENSOR_SHAPE, TURN_LAYER, POLICY_SIZE, PROMOTION_AWARE_POLICY_SIZE,
    POLICY_LOSS_WEIGHT,
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    VALUE_TARGET,
    VALUE_HEAD_MODE, WDL_LOSS_WEIGHT, WDL_DRAW_EPSILON,
    PROCESSED_DATA_DIR, MODEL_DIR,
    VALUE_LOSS_EXPONENT, LR_GAMMA, RANDOM_SEED,
    WEIGHT_DECAY, GRAD_CLIP_NORM, WARMUP_EPOCHS, WARMUP_START_FACTOR,
    POLICY_HEAD_CHANNELS, POLICY_HEAD_TYPE, POLICY_ATTENTION_CHANNELS,
    SIDE_POLICY_ADAPTERS,
    STEM_CHANNELS, RESIDUAL_BLOCK_CHANNELS,
    USE_SE_BLOCKS, SE_REDUCTION,
    SPATIAL_VALUE_HEAD, VALUE_HEAD_CONV_CHANNELS,
    USE_MOVES_LEFT_HEAD, MOVES_LEFT_HEAD_CHANNELS, MOVES_LEFT_LOSS_WEIGHT,
)

# Input channels = last dim of the current position encoding.
IN_CHANNELS = TENSOR_SHAPE[2]


def set_seed(seed):
    """Seed Python/NumPy/Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_git_commit():
    """Best-effort short git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def build_optimizer(model, lr, weight_decay):
    """AdamW with weight decay excluded for norm layers and biases."""
    norm_layers = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )
    decay_params = []
    no_decay_params = []

    for _, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param_name == "bias" or isinstance(module, norm_layers):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    return optimizer, len(decay_params), len(no_decay_params)


class ModelEMA:
    """Exponential moving average of parameters and floating-point buffers."""
    def __init__(self, model, decay):
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            source = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source)


def _wdl_expectation_from_logits(logits):
    """Convert WDL logits [loss, draw, win] to scalar expectation in [-1, 1]."""
    probs = torch.softmax(logits, dim=1)
    # expected value from side-to-move perspective: P(win) - P(loss)
    return (probs[:, 2:3] - probs[:, 0:1])


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False, se_reduction=SE_REDUCTION):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SqueezeExcite(out_ch, reduction=se_reduction) if use_se else None
        # 1x1 projection if channel count changes
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        shortcut = x if self.proj is None else self.proj(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.se is not None:
            x = self.se(x)
        return F.relu(x + shortcut)


class DualHeadNet(nn.Module):
    """ResNet with a value head and a policy head.

    Shared backbone:
      Configurable stem and residual tower from config.py
    Value head (scalar, tanh), one of:
      GAP     -> Dense 128 -> Dense 64 -> Dense 1        (default)
      Conv Cvx1x1 -> BN -> ReLU -> Flatten -> Dense 256 -> Dense 1
                                                        (spatial_value_head)
    Optional WDL head (use_wdl_head), read as P(win) - P(loss).
    Policy head:
      Conv Cx1x1 -> BN -> ReLU -> Flatten -> Dense 4096 (logits)
    """
    def __init__(
        self,
        input_channels=IN_CHANNELS,
        policy_head_channels=POLICY_HEAD_CHANNELS,
        policy_head_type=POLICY_HEAD_TYPE,
        policy_attention_channels=POLICY_ATTENTION_CHANNELS,
        side_policy_adapters=SIDE_POLICY_ADAPTERS,
        promotion_policy=False,
        stem_channels=STEM_CHANNELS,
        residual_block_channels=RESIDUAL_BLOCK_CHANNELS,
        use_se_blocks=USE_SE_BLOCKS,
        se_reduction=SE_REDUCTION,
        use_wdl_head=False,
        value_head_mode=VALUE_HEAD_MODE,
        spatial_value_head=SPATIAL_VALUE_HEAD,
        value_head_conv_channels=VALUE_HEAD_CONV_CHANNELS,
        use_moves_left_head=USE_MOVES_LEFT_HEAD,
        moves_left_head_channels=MOVES_LEFT_HEAD_CHANNELS,
    ):
        super().__init__()
        self.spatial_value_head = bool(spatial_value_head)
        self.value_head_conv_channels = int(value_head_conv_channels)
        self.input_channels = int(input_channels)
        self.policy_head_channels = int(policy_head_channels)
        self.policy_head_type = str(policy_head_type)
        self.policy_attention_channels = int(policy_attention_channels)
        self.side_policy_adapters = bool(side_policy_adapters)
        self.promotion_policy = bool(promotion_policy)
        self.policy_output_size = (PROMOTION_AWARE_POLICY_SIZE
                                   if self.promotion_policy else POLICY_SIZE)
        self.stem_channels = int(stem_channels)
        self.residual_block_channels = tuple(int(c) for c in residual_block_channels)
        self.use_se_blocks = bool(use_se_blocks)
        self.se_reduction = int(se_reduction)
        self.use_wdl_head = bool(use_wdl_head)
        self.value_head_mode = str(value_head_mode)
        self.use_moves_left_head = bool(use_moves_left_head)
        self.moves_left_head_channels = int(moves_left_head_channels)
        if not self.residual_block_channels:
            raise ValueError("residual_block_channels must contain at least one block")
        if self.se_reduction <= 0:
            raise ValueError("se_reduction must be > 0")
        if self.value_head_conv_channels <= 0:
            raise ValueError("value_head_conv_channels must be > 0")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        if self.policy_head_type not in ("dense", "attention"):
            raise ValueError("policy_head_type must be 'dense' or 'attention'")
        if self.policy_attention_channels <= 0:
            raise ValueError("policy_attention_channels must be > 0")
        if self.side_policy_adapters and self.policy_head_type != "attention":
            raise ValueError("side_policy_adapters requires the attention policy head")
        if self.moves_left_head_channels <= 0:
            raise ValueError("moves_left_head_channels must be > 0")
        if self.value_head_mode not in ("scalar", "wdl"):
            raise ValueError("value_head_mode must be 'scalar' or 'wdl'")
        if self.value_head_mode == "wdl" and not self.use_wdl_head:
            raise ValueError(f"value_head_mode='{self.value_head_mode}' requires use_wdl_head=True")
        if self.use_wdl_head and self.value_head_mode == "scalar":
            # A state_dict otherwise cannot distinguish an auxiliary WDL head
            # from a WDL head used as the engine's primary value output.
            self.register_buffer("_aux_wdl_head", torch.tensor(1, dtype=torch.uint8))

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, self.stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.stem_channels),
            nn.ReLU(),
        )
        # Residual tower. Keep deterministic resN naming for checkpoint compatibility.
        in_ch = self.stem_channels
        for i, out_ch in enumerate(self.residual_block_channels, start=1):
            setattr(
                self,
                f"res{i}",
                ResidualBlock(
                    in_ch, out_ch,
                    use_se=self.use_se_blocks,
                    se_reduction=self.se_reduction,
                ),
            )
            in_ch = out_ch
        self.residual_block_count = len(self.residual_block_channels)
        self.backbone_out_channels = in_ch

        # Value head(s)
        self.value_head = self._make_value_head()
        if self.use_wdl_head:
            self.wdl_head = self._make_wdl_head()
        if self.use_moves_left_head:
            self.moves_left_head = self._make_moves_left_head()

        # Policy head
        if self.policy_head_type == "dense":
            self.policy_conv, self.policy_fc = self._make_policy_head()
        else:
            self.policy_attention = nn.Conv2d(
                self.backbone_out_channels,
                2 * self.policy_attention_channels + 2,
                1,
                bias=True,
            )
            if self.side_policy_adapters:
                # Any pair of side-specific projections can be represented as
                # shared +/- delta. Zero initialization starts from the exact
                # shared-head geometry while allowing White and Black gradients
                # to immediately separate the residual projection.
                self.policy_side_adapter = nn.Conv2d(
                    self.backbone_out_channels,
                    2 * self.policy_attention_channels + 2,
                    1,
                    bias=True,
                )
                nn.init.zeros_(self.policy_side_adapter.weight)
                nn.init.zeros_(self.policy_side_adapter.bias)
            # LC0's attention head includes learned move biases. This compact
            # table lets the net represent board geometry (source,destination)
            # without recreating the 8.4M-parameter dense layer.
            self.policy_relative_bias = nn.Parameter(torch.zeros(64, 64))
        if self.promotion_policy:
            # Twelve position-dependent deltas per source square:
            # three destination directions x q/r/b/n.  Only the White rank-7
            # and Black rank-2 source rows are read.  Zero initialization makes
            # a lifted legacy checkpoint reproduce its old source/destination
            # logits exactly until promotion-specific evidence is learned.
            self.policy_promotion_delta = nn.Conv2d(
                self.backbone_out_channels, 12, 1, bias=True)
            nn.init.zeros_(self.policy_promotion_delta.weight)
            nn.init.zeros_(self.policy_promotion_delta.bias)
            self.register_buffer(
                "_promotion_base_indices",
                self._make_promotion_base_indices(),
                persistent=False,
            )

    @staticmethod
    def _make_promotion_base_indices():
        indices = []
        for from_rank, to_rank in ((6, 7), (1, 0)):
            for from_file in range(8):
                from_sq = from_rank * 8 + from_file
                for direction in (-1, 0, 1):
                    # Off-board cells are never legal or supervised, but still
                    # occupy stable ABI slots.  Clamp only their unused base.
                    to_file = min(7, max(0, from_file + direction))
                    to_sq = to_rank * 8 + to_file
                    indices.extend([from_sq * 64 + to_sq] * 4)
        return torch.tensor(indices, dtype=torch.long)

    def _make_value_head(self):
        # No dropout (REWORK_PLAN.md Phase 2.2): heavy dropout on a small GAP feature
        # pushes predictions toward the batch mean (~0), which was a direct contributor
        # to flat value calibration.  Weight decay (build_optimizer) is the regularizer.
        if self.spatial_value_head:
            # Keeps the 8x8 layout instead of averaging it away. Index 0 is a
            # Conv2d (the GAP variant's index 0 is a parameterless pool), so
            # "value_head.0.weight" in a state dict identifies this variant.
            return nn.Sequential(
                nn.Conv2d(self.backbone_out_channels,
                          self.value_head_conv_channels, 1, bias=False),
                nn.BatchNorm2d(self.value_head_conv_channels),
                nn.ReLU(),
                nn.Flatten(),                                  # (N, Cv*64)
                nn.Linear(self.value_head_conv_channels * 64, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Tanh(),
            )
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (N, C, 1, 1)
            nn.Flatten(),                  # (N, C)
            nn.Linear(self.backbone_out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def _make_wdl_head(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone_out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # [loss, draw, win]
        )

    def _make_moves_left_head(self):
        """Small LC0-style auxiliary head; output is remaining decisions."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone_out_channels, self.moves_left_head_channels),
            nn.ReLU(),
            nn.Linear(self.moves_left_head_channels, 1),
            nn.Softplus(),
        )

    def _make_policy_head(self):
        conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, self.policy_head_channels, 1, bias=False),
            nn.BatchNorm2d(self.policy_head_channels),
            nn.ReLU(),
        )
        fc = nn.Linear(self.policy_head_channels * 8 * 8, POLICY_SIZE)
        return conv, fc

    def _base_policy_logits(self, backbone, side_turn):
        if self.policy_head_type == "dense":
            p = self.policy_conv(backbone)
            p = p.flatten(1)
            return self.policy_fc(p)

        projected = self.policy_attention(backbone)
        if self.side_policy_adapters:
            side = side_turn.to(dtype=projected.dtype).view(-1, 1, 1, 1)
            projected = projected + side * self.policy_side_adapter(backbone)
        projected = projected.flatten(2).transpose(1, 2)
        width = self.policy_attention_channels
        query = projected[:, :, :width]
        key = projected[:, :, width:2 * width]
        source_bias = projected[:, :, 2 * width]
        destination_bias = projected[:, :, 2 * width + 1]
        logits = torch.bmm(query, key.transpose(1, 2)) / (width ** 0.5)
        logits = logits + source_bias.unsqueeze(2) + destination_bias.unsqueeze(1)
        logits = logits + self.policy_relative_bias.unsqueeze(0)
        return logits.flatten(1)

    def _policy_logits(self, backbone, side_turn):
        base_logits = self._base_policy_logits(backbone, side_turn)
        if not self.promotion_policy:
            return base_logits
        delta = self.policy_promotion_delta(backbone)
        white = delta[:, :, 6, :].permute(0, 2, 1).reshape(-1, 96)
        black = delta[:, :, 1, :].permute(0, 2, 1).reshape(-1, 96)
        promotion_delta = torch.cat((white, black), dim=1)
        promotion_base = base_logits.index_select(
            1, self._promotion_base_indices.to(base_logits.device))
        return torch.cat((base_logits, promotion_base + promotion_delta), dim=1)

    def _forward_backbone(self, x):
        # x: (N, C, 8, 8)  — PyTorch uses channels-first
        side_turn = x[:, TURN_LAYER, 0, 0]
        x = self.stem(x)
        for i in range(1, self.residual_block_count + 1):
            x = getattr(self, f"res{i}")(x)
        return x, side_turn

    def _compute_heads(self, backbone, side_turn):
        scalar_value = self.value_head(backbone)
        policy = self._policy_logits(backbone, side_turn)
        wdl_logits = self.wdl_head(backbone) if self.use_wdl_head else None

        value = scalar_value
        if self.use_wdl_head:
            wdl_value = _wdl_expectation_from_logits(wdl_logits)
            if self.value_head_mode == "wdl":
                value = wdl_value

        return value, policy, wdl_logits

    def forward(self, x):
        backbone, side_turn = self._forward_backbone(x)
        value, policy, _ = self._compute_heads(backbone, side_turn)
        return value, policy

    def forward_with_wdl(self, x):
        backbone, side_turn = self._forward_backbone(x)
        value, policy, wdl_logits = self._compute_heads(backbone, side_turn)
        return value, policy, wdl_logits

    def forward_with_aux(self, x):
        """Training-only superset; the ordinary inference ABI stays 2-headed."""
        backbone, side_turn = self._forward_backbone(x)
        value, policy, wdl_logits = self._compute_heads(backbone, side_turn)
        moves_left = (self.moves_left_head(backbone)
                      if self.use_moves_left_head else None)
        return value, policy, wdl_logits, moves_left



def infer_policy_head_channels(state_dict):
    """Infer policy bottleneck width from checkpoint state dict."""
    w = state_dict.get("policy_conv.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[0] > 0:
        return int(w.shape[0])
    return POLICY_HEAD_CHANNELS


def infer_policy_head_config(state_dict):
    """Infer dense vs compact attention policy geometry from a checkpoint."""
    attention = state_dict.get("policy_attention.weight")
    if (isinstance(attention, torch.Tensor) and attention.ndim == 4
            and attention.shape[0] >= 4):
        width = (int(attention.shape[0]) - 2) // 2
        return "attention", POLICY_HEAD_CHANNELS, width
    return "dense", infer_policy_head_channels(state_dict), POLICY_ATTENTION_CHANNELS


def infer_side_policy_adapters(state_dict):
    """Infer the optional side-conditioned residual attention projection."""
    return any(key.startswith("policy_side_adapter.") for key in state_dict)


def infer_promotion_policy(state_dict):
    """Infer the optional distinct-promotion policy extension."""
    return any(key.startswith("policy_promotion_delta.") for key in state_dict)


def infer_input_channels(state_dict):
    """Infer the position encoding width from the checkpoint stem."""
    w = state_dict.get("stem.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[1] > 0:
        return int(w.shape[1])
    return IN_CHANNELS


def infer_spatial_value_head_config(state_dict):
    """Infer whether the value head keeps spatial layout, and its width.

    The spatial head begins with a Conv2d; the GAP head begins with a
    parameterless AdaptiveAvgPool2d. So a 4-D ``value_head.0.weight`` is
    present for one variant and absent for the other — an unambiguous marker
    that needs no extra flag persisted in the checkpoint.
    """
    w = state_dict.get("value_head.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[0] > 0:
        return True, int(w.shape[0])
    return False, VALUE_HEAD_CONV_CHANNELS


def infer_wdl_head_config(state_dict):
    """Infer whether WDL head exists in checkpoint."""
    has_wdl = any(
        k.startswith("wdl_head.")
        for k in state_dict.keys()
    )
    if not has_wdl:
        return False, "scalar"
    if "_aux_wdl_head" in state_dict:
        return True, "scalar"
    return True, "wdl"


def infer_moves_left_head_config(state_dict):
    """Infer optional moves-left head presence and hidden width."""
    w = state_dict.get("moves_left_head.2.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] > 0:
        return True, int(w.shape[0])
    return False, MOVES_LEFT_HEAD_CHANNELS


def infer_backbone_architecture(state_dict):
    """Infer stem width and residual tower channels from a checkpoint."""
    stem_channels = STEM_CHANNELS
    stem_w = state_dict.get("stem.0.weight")
    if isinstance(stem_w, torch.Tensor) and stem_w.ndim == 4 and stem_w.shape[0] > 0:
        stem_channels = int(stem_w.shape[0])

    pattern = re.compile(r"^res(\d+)\.conv1\.weight$")
    block_out_channels = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            continue
        m = pattern.match(key)
        if m:
            idx = int(m.group(1))
            block_out_channels[idx] = int(tensor.shape[0])

    if block_out_channels:
        block_channels = tuple(block_out_channels[i] for i in sorted(block_out_channels))
    else:
        block_channels = RESIDUAL_BLOCK_CHANNELS
    return stem_channels, block_channels


def infer_se_config(state_dict):
    """Infer whether SE blocks are present (and reduction) from checkpoint."""
    se_key = None
    for k in state_dict.keys():
        if ".se.fc1.weight" in k:
            se_key = k
            break
    if se_key is None:
        return False, SE_REDUCTION
    w = state_dict.get(se_key)
    if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] > 0:
        channels = int(w.shape[1])
        hidden = int(w.shape[0])
        reduction = max(1, channels // hidden)
    else:
        reduction = SE_REDUCTION
    return True, reduction


def build_model(
    input_channels=IN_CHANNELS,
    policy_head_channels=POLICY_HEAD_CHANNELS,
    policy_head_type=POLICY_HEAD_TYPE,
    policy_attention_channels=POLICY_ATTENTION_CHANNELS,
    side_policy_adapters=SIDE_POLICY_ADAPTERS,
    promotion_policy=False,
    stem_channels=STEM_CHANNELS,
    residual_block_channels=RESIDUAL_BLOCK_CHANNELS,
    use_se_blocks=USE_SE_BLOCKS,
    se_reduction=SE_REDUCTION,
    use_wdl_head=False,
    value_head_mode=VALUE_HEAD_MODE,
    spatial_value_head=SPATIAL_VALUE_HEAD,
    value_head_conv_channels=VALUE_HEAD_CONV_CHANNELS,
    use_moves_left_head=USE_MOVES_LEFT_HEAD,
    moves_left_head_channels=MOVES_LEFT_HEAD_CHANNELS,
):
    return DualHeadNet(
        input_channels=input_channels,
        policy_head_channels=policy_head_channels,
        policy_head_type=policy_head_type,
        policy_attention_channels=policy_attention_channels,
        side_policy_adapters=side_policy_adapters,
        promotion_policy=promotion_policy,
        stem_channels=stem_channels,
        residual_block_channels=residual_block_channels,
        use_se_blocks=use_se_blocks,
        se_reduction=se_reduction,
        use_wdl_head=use_wdl_head,
        value_head_mode=value_head_mode,
        spatial_value_head=spatial_value_head,
        value_head_conv_channels=value_head_conv_channels,
        use_moves_left_head=use_moves_left_head,
        moves_left_head_channels=moves_left_head_channels,
    )


def load_model_for_inference(checkpoint_path, device):
    """Load model with architecture inferred from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    input_channels = infer_input_channels(state_dict)
    policy_head_type, pol_ch, attention_ch = infer_policy_head_config(state_dict)
    side_policy_adapters = infer_side_policy_adapters(state_dict)
    promotion_policy = infer_promotion_policy(state_dict)
    stem_ch, block_ch = infer_backbone_architecture(state_dict)
    use_se_blocks, se_reduction = infer_se_config(state_dict)
    use_wdl_head, value_head_mode = infer_wdl_head_config(state_dict)
    spatial_value_head, value_conv_ch = infer_spatial_value_head_config(state_dict)
    use_moves_left_head, moves_left_ch = infer_moves_left_head_config(state_dict)
    model = build_model(
        input_channels=input_channels,
        policy_head_channels=pol_ch,
        policy_head_type=policy_head_type,
        policy_attention_channels=attention_ch,
        side_policy_adapters=side_policy_adapters,
        promotion_policy=promotion_policy,
        stem_channels=stem_ch,
        residual_block_channels=block_ch,
        use_se_blocks=use_se_blocks,
        se_reduction=se_reduction,
        use_wdl_head=use_wdl_head,
        value_head_mode=value_head_mode,
        spatial_value_head=spatial_value_head,
        value_head_conv_channels=value_conv_ch,
        use_moves_left_head=use_moves_left_head,
        moves_left_head_channels=moves_left_ch,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, pol_ch


def load_state_dict_flexible(model, state_dict):
    """Load only shape-compatible tensors. Returns (loaded_count, skipped_keys)."""
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return len(compatible), skipped


def load_data(data_dir, include_moves_left=False, include_legal_masks=False,
              include_capture_results=False):
    """Load processed training data and splits.

    The legacy 7-item return stays unchanged unless the optional auxiliary
    arrays are requested.
    """
    positions = np.load(os.path.join(data_dir, "positions.npy"))
    mcts_values = np.load(os.path.join(data_dir, "mcts_values.npy"))
    game_results = np.load(os.path.join(data_dir, "game_results.npy"))
    policies = np.load(os.path.join(data_dir, "policies.npy"))
    policy_weights_path = os.path.join(data_dir, "policy_weights.npy")
    if os.path.exists(policy_weights_path):
        policy_weights = np.load(policy_weights_path)
    else:
        # Backward compatibility with processed corpora created before policy
        # masking existed: every record remains a policy teacher.
        policy_weights = np.ones((len(policies),), dtype=np.float32)
    value_weights_path = os.path.join(data_dir, "value_weights.npy")
    if os.path.exists(value_weights_path):
        value_weights = np.load(value_weights_path)
    else:
        # Corpora processed before value weighting existed: every record is a
        # value teacher, which is exactly what they trained as.
        value_weights = np.ones((len(policies),), dtype=np.float32)
    with np.load(os.path.join(data_dir, "splits.npz")) as split_file:
        splits = {name: split_file[name] for name in split_file.files}
    base = (positions, mcts_values, game_results, policies, policy_weights,
            value_weights, splits)
    if include_moves_left:
        moves_path = os.path.join(data_dir, "moves_left.npy")
        moves_weights_path = os.path.join(data_dir, "moves_left_weights.npy")
        if not os.path.exists(moves_path) or not os.path.exists(moves_weights_path):
            raise FileNotFoundError(
                "moves-left training requested, but processed corpus lacks "
                "moves_left.npy/moves_left_weights.npy; re-run data_processor.py")
        moves_left = np.load(moves_path)
        moves_left_weights = np.load(moves_weights_path)
        if (len(moves_left) != len(policies)
                or len(moves_left_weights) != len(policies)):
            raise ValueError("moves-left arrays are not aligned with policies.npy")
        base = base + (moves_left, moves_left_weights)
    if include_legal_masks:
        masks_path = os.path.join(data_dir, "legal_masks_packed.npy")
        if not os.path.exists(masks_path):
            raise FileNotFoundError(
                "legal policy masking requested, but processed corpus lacks "
                "legal_masks_packed.npy; re-run data_processor.py")
        legal_masks = np.load(masks_path)
        expected_mask_bytes = (int(policies.shape[1]) + 7) // 8
        if legal_masks.shape != (len(policies), expected_mask_bytes):
            raise ValueError(
                "legal_masks_packed.npy must have shape "
                f"({len(policies)}, {expected_mask_bytes}), got {legal_masks.shape}")
        base = base + (legal_masks,)
    if include_capture_results:
        capture_path = os.path.join(data_dir, "capture_results.npy")
        if not os.path.exists(capture_path):
            raise FileNotFoundError(
                "capture-result training requested, but processed corpus lacks "
                "capture_results.npy; re-run data_processor.py")
        capture_results = np.load(capture_path)
        if capture_results.shape != (len(policies),):
            raise ValueError(
                "capture_results.npy must align with policies.npy; got "
                f"{capture_results.shape} for {len(policies)} rows")
        base = base + (capture_results,)
    return base


def to_side_perspective(values_white_perspective, positions):
    """Convert white-perspective targets to side-to-move perspective."""
    side_sign = np.where(positions[:, 0, 0, TURN_LAYER] > 0, 1.0, -1.0).astype(np.float32)
    return values_white_perspective * side_sign


def apply_black_policy_weight(policy_weights, positions, black_weight=1.0):
    """Return policy weights with Black-to-move examples relatively scaled.

    The weighted CE already normalizes by total enabled weight, so this changes
    the White/Black mixture without changing labels or the overall loss scale.
    """
    black_weight = float(black_weight)
    if black_weight <= 0:
        raise ValueError("black_weight must be > 0")
    adjusted = np.asarray(policy_weights, dtype=np.float32).copy()
    black_turn = positions[:, 0, 0, TURN_LAYER] <= 0
    adjusted[black_turn] *= black_weight
    return adjusted


def configure_policy_head_only(model, enabled=False):
    """Optionally freeze everything except the compact policy head."""
    if not enabled:
        return [name for name, parameter in model.named_parameters()
                if parameter.requires_grad]
    trainable = []
    for name, parameter in model.named_parameters():
        keep = (name == "policy_relative_bias"
                or name.startswith("policy_attention.")
                or name.startswith("policy_side_adapter.")
                or name.startswith("policy_promotion_delta."))
        parameter.requires_grad_(keep)
        if keep:
            trainable.append(name)
    if not trainable:
        raise ValueError("policy-head-only training requires an attention policy head")
    return trainable


def configure_promotion_head_only(model, enabled=False):
    """Freeze the lifted v20 model except for promotion-choice deltas."""
    if not enabled:
        return [name for name, parameter in model.named_parameters()
                if parameter.requires_grad]
    if not model.promotion_policy:
        raise ValueError("promotion-head-only training requires promotion policy")
    model._freeze_backbone_batchnorm = True
    trainable = []
    for name, parameter in model.named_parameters():
        keep = name.startswith("policy_promotion_delta.")
        parameter.requires_grad_(keep)
        if keep:
            trainable.append(name)
    return trainable


def _set_training_mode(model):
    """Enter training mode without mutating frozen backbone BN statistics."""
    model.train()
    if getattr(model, "_freeze_backbone_batchnorm", False):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def get_targets(mcts_values, game_results, target_type):
    """Build value training targets based on the chosen strategy."""
    if target_type in ("game_result", "capture_result"):
        return game_results
    elif target_type == "mcts_value":
        return mcts_values
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def build_wdl_targets(values, draw_epsilon=WDL_DRAW_EPSILON):
    """Build class labels [loss=0, draw=1, win=2] from scalar targets."""
    eps = float(draw_epsilon)
    if eps < 0:
        raise ValueError("draw_epsilon must be >= 0")
    labels = np.ones((len(values),), dtype=np.int64)
    labels[values > eps] = 2
    labels[values < -eps] = 0
    return labels


def _make_loader(X, y_val, y_pol, batch_size, shuffle=True, generator=None,
                 y_wdl=None, y_policy_weight=None, y_value_weight=None,
                 y_moves_left=None, y_moves_left_weight=None,
                 y_legal_masks_packed=None):
    """Create a DataLoader from numpy arrays.

    Transposes X from (N, 8, 8, C) to (N, C, 8, 8) for PyTorch.
    """
    X_t = torch.from_numpy(X.transpose(0, 3, 1, 2))  # channels-first
    y_v = torch.from_numpy(y_val).unsqueeze(1)         # (N, 1)
    y_p = torch.from_numpy(y_pol)                      # (N, 4096)
    if y_policy_weight is None:
        y_policy_weight = np.ones((len(y_pol),), dtype=np.float32)
    y_pw = torch.from_numpy(y_policy_weight).float()
    if y_value_weight is None:
        y_value_weight = np.ones((len(y_pol),), dtype=np.float32)
    y_vw = torch.from_numpy(y_value_weight).float()
    parts = [X_t, y_v, y_p, y_pw, y_vw]
    if y_wdl is not None:
        parts.append(torch.from_numpy(y_wdl).long())
    if y_moves_left is not None:
        if y_moves_left_weight is None:
            y_moves_left_weight = np.ones(
                (len(y_moves_left),), dtype=np.float32)
        y_ml = torch.from_numpy(y_moves_left).float().unsqueeze(1)
        y_mlw = torch.from_numpy(y_moves_left_weight).float()
        parts.extend((y_ml, y_mlw))
    if y_legal_masks_packed is not None:
        parts.append(torch.from_numpy(y_legal_masks_packed).to(torch.uint8))
    ds = TensorDataset(*parts)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0, generator=generator)


def _unpack_loader_batch(batch):
    """Normalize current and legacy batches -> (X, y_val, y_pol, y_pw, y_vw, y_wdl).

    Batches have grown twice (policy weights, then value weights) and older
    shapes still reach here from tests and from any caller building its own
    TensorDataset. Length alone is ambiguous at 5 -- it is either the current
    (X, v, p, pw, vw) or the legacy (X, v, p, pw, wdl) -- so the tie is broken
    on dtype: weights are float, WDL labels are long.
    """
    ones = lambda n: torch.ones((n,), dtype=torch.float32)  # noqa: E731

    if len(batch) == 6:
        return batch
    if len(batch) == 5:
        X_b, yv_b, yp_b, ypw_b, fifth = batch
        if fifth.dtype.is_floating_point:
            return X_b, yv_b, yp_b, ypw_b, fifth, None
        return X_b, yv_b, yp_b, ypw_b, ones(len(X_b)), fifth
    if len(batch) == 4:
        X_b, yv_b, yp_b, fourth = batch
        if fourth.dtype.is_floating_point:
            return X_b, yv_b, yp_b, fourth, ones(len(X_b)), None
        return X_b, yv_b, yp_b, ones(len(X_b)), ones(len(X_b)), fourth
    if len(batch) == 3:
        X_b, yv_b, yp_b = batch
        return X_b, yv_b, yp_b, ones(len(X_b)), ones(len(X_b)), None
    raise ValueError(f"Unexpected batch tuple length: {len(batch)}")


def _unpack_aux_loader_batch(batch, use_moves_left_head=False,
                             use_wdl_head=False,
                             use_legal_policy_mask=False):
    """Extend the stable legacy unpacker with opt-in auxiliary tensors."""
    if not use_moves_left_head and not use_legal_policy_mask:
        # DataLoader currently collates TensorDataset samples into a list,
        # while tests and external callers commonly provide tuples.
        return tuple(_unpack_loader_batch(batch)) + (None, None, None)
    if len(batch) < 5:
        raise ValueError("Auxiliary loaders require policy and value weights")
    X_b, yv_b, yp_b, ypw_b, yvw_b = batch[:5]
    cursor = 5
    yw_b = None
    if use_wdl_head:
        yw_b = batch[cursor]
        cursor += 1
    yml_b = ymlw_b = None
    if use_moves_left_head:
        yml_b, ymlw_b = batch[cursor:cursor + 2]
        cursor += 2
    legal_b = None
    if use_legal_policy_mask:
        legal_b = batch[cursor]
        cursor += 1
    if cursor != len(batch):
        raise ValueError(
            f"Auxiliary loader consumed {cursor} of {len(batch)} tensors")
    return X_b, yv_b, yp_b, ypw_b, yvw_b, yw_b, yml_b, ymlw_b, legal_b


def _power_loss(pred, target, exponent=VALUE_LOSS_EXPONENT, weights=None):
    """Power-law loss: mean(|pred - target|^exp). Stockfish uses 2.5.

    With weights, the mean is taken over the weighted records and normalized
    by total weight -- so an all-ones weight vector reproduces the unweighted
    mean exactly, and a zero-weight record contributes no value gradient.
    Mirrors weighted_policy_cross_entropy.
    """
    losses = torch.pow(torch.abs(pred - target), exponent)
    if weights is None:
        return losses.mean()
    losses = losses.reshape(-1)
    w = weights.to(device=losses.device, dtype=losses.dtype).reshape(-1)
    total = w.sum()
    if total.item() <= 0:
        # No value teachers in this batch: contribute nothing, but keep the
        # graph connected so .backward() does not fail.
        return losses.sum() * 0.0
    return (losses * w).sum() / total


def _weighted_wdl_ce(logits, labels, weights):
    """WDL cross-entropy under value weights.

    The WDL head is a value head, so a record with value_weight 0 must not
    teach through it either -- otherwise "policy only" would leak beliefs in
    by the back door whenever --value-head wdl is used.
    """
    losses = F.cross_entropy(logits, labels, reduction="none")
    w = weights.to(device=losses.device, dtype=losses.dtype).reshape(-1)
    total = w.sum()
    if total.item() <= 0:
        return losses.sum() * 0.0
    return (losses * w).sum() / total


def unpack_legal_policy_mask(packed, policy_size=POLICY_SIZE):
    """Unpack NumPy-compatible big-endian packbits on the current device."""
    if packed.ndim != 2 or packed.shape[1] * 8 < policy_size:
        raise ValueError("packed legal mask has the wrong shape")
    shifts = torch.arange(7, -1, -1, device=packed.device,
                          dtype=torch.uint8)
    bits = ((packed.to(torch.uint8).unsqueeze(-1) >> shifts) & 1)
    return bits.reshape(packed.shape[0], -1)[:, :policy_size].bool()


def mask_policy_logits(logits, legal_masks_packed):
    """Set illegal policy logits to the dtype minimum before softmax/argmax."""
    legal = unpack_legal_policy_mask(legal_masks_packed, logits.shape[1])
    if not torch.all(legal.any(dim=1)):
        raise ValueError("legal policy mask contains a row with no legal moves")
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min), legal


def mask_inactive_promotion_logits(logits, targets):
    """Hide extension logits on rows with no promotion action target."""
    if logits.shape[1] != PROMOTION_AWARE_POLICY_SIZE:
        return logits
    inactive = targets[:, POLICY_SIZE:].sum(dim=1) <= 0
    if not torch.any(inactive):
        return logits
    active_mask = torch.ones_like(logits, dtype=torch.bool)
    active_mask[inactive, POLICY_SIZE:] = False
    return logits.masked_fill(~active_mask, torch.finfo(logits.dtype).min)


def weighted_policy_cross_entropy(logits, targets, weights,
                                  legal_masks_packed=None):
    """Soft-target policy CE normalized over enabled, optionally legal moves."""
    if legal_masks_packed is not None:
        logits, legal = mask_policy_logits(logits, legal_masks_packed)
        illegal_mass = targets.masked_fill(legal, 0).sum(dim=1)
        enabled = weights.reshape(-1) > 0
        if torch.any(illegal_mass[enabled] > 1e-5):
            raise ValueError("policy target assigns mass to an illegal move")
    elif logits.shape[1] == PROMOTION_AWARE_POLICY_SIZE:
        # On ordinary positions the extension has no semantic legal action.
        # Excluding it makes a lifted model's loss exactly the legacy 4096-way
        # loss, so 98%+ ordinary rows cannot train the small delta head merely
        # to suppress impossible promotions.
        logits = mask_inactive_promotion_logits(logits, targets)
    losses = F.cross_entropy(logits, targets, reduction="none")
    weights = weights.to(device=losses.device, dtype=losses.dtype).reshape(-1)
    total_weight = weights.sum()
    if total_weight.item() <= 0:
        return losses.sum() * 0.0
    return (losses * weights).sum() / total_weight


def weighted_moves_left_huber(predictions, targets, weights):
    """LC0-style robust regression, masked to trusted decisive trajectories."""
    losses = F.smooth_l1_loss(predictions, targets, reduction="none").reshape(-1)
    w = weights.to(device=losses.device, dtype=losses.dtype).reshape(-1)
    total = w.sum()
    if total.item() <= 0:
        return losses.sum() * 0.0
    return (losses * w).sum() / total


def _set_epoch_lr(optimizer, epoch, base_lr, warmup_epochs, warmup_start_factor):
    """Apply linear warmup schedule and return the LR used this epoch."""
    if warmup_epochs <= 0:
        return optimizer.param_groups[0]["lr"]

    if epoch <= warmup_epochs:
        if warmup_epochs == 1:
            t = 1.0
        else:
            t = (epoch - 1) / (warmup_epochs - 1)
        lr = base_lr * (warmup_start_factor + (1.0 - warmup_start_factor) * t)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    if epoch == warmup_epochs + 1:
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr
    return optimizer.param_groups[0]["lr"]


def _train_epoch(model, loader, optimizer, device, policy_weight, grad_clip_norm,
                  use_wdl_head=False, wdl_loss_weight=0.0,
                  value_head_mode="scalar", use_moves_left_head=False,
                  moves_left_loss_weight=0.0, use_legal_policy_mask=False,
                  ema=None):
    _set_training_mode(model)
    total_loss = 0.0
    total_val_loss = 0.0
    total_pol_loss = 0.0
    total_wdl_loss = 0.0
    total_moves_left_loss = 0.0
    n = 0

    for batch in loader:
        (X_b, yv_b, yp_b, ypw_b, yvw_b, yw_b,
         yml_b, ymlw_b, legal_b) = _unpack_aux_loader_batch(
            batch, use_moves_left_head=use_moves_left_head,
            use_wdl_head=use_wdl_head,
            use_legal_policy_mask=use_legal_policy_mask)
        X_b = X_b.to(device)
        yv_b = yv_b.to(device)
        yp_b = yp_b.to(device)
        ypw_b = ypw_b.to(device)
        yvw_b = yvw_b.to(device)
        if yw_b is not None:
            yw_b = yw_b.to(device)
        if yml_b is not None:
            yml_b = yml_b.to(device)
            ymlw_b = ymlw_b.to(device)
        if legal_b is not None:
            legal_b = legal_b.to(device)

        if use_moves_left_head:
            value_pred, policy_pred, wdl_logits, moves_left_pred = (
                model.forward_with_aux(X_b))
        elif use_wdl_head:
            value_pred, policy_pred, wdl_logits = model.forward_with_wdl(X_b)
            moves_left_pred = None
        else:
            value_pred, policy_pred = model(X_b)
            wdl_logits = None
            moves_left_pred = None
        value_loss_pred = value_pred
        loss_val = _power_loss(value_loss_pred, yv_b, weights=yvw_b)
        loss_pol = weighted_policy_cross_entropy(
            policy_pred, yp_b, ypw_b, legal_masks_packed=legal_b)
        loss = loss_val + policy_weight * loss_pol
        loss_wdl = torch.zeros((), device=device)
        if (
            use_wdl_head
            and wdl_logits is not None
            and yw_b is not None
            and wdl_loss_weight > 0
        ):
            loss_wdl = _weighted_wdl_ce(wdl_logits, yw_b, yvw_b)
            loss = loss + wdl_loss_weight * loss_wdl
        loss_moves_left = torch.zeros((), device=device)
        if (use_moves_left_head and moves_left_pred is not None
                and yml_b is not None and moves_left_loss_weight > 0):
            loss_moves_left = weighted_moves_left_huber(
                moves_left_pred, yml_b, ymlw_b)
            loss = loss + moves_left_loss_weight * loss_moves_left

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = X_b.size(0)
        total_loss += loss.item() * bs
        total_val_loss += loss_val.item() * bs
        total_pol_loss += loss_pol.item() * bs
        total_wdl_loss += loss_wdl.item() * bs
        total_moves_left_loss += loss_moves_left.item() * bs
        n += bs

    return (
        total_loss / n,
        total_val_loss / n,
        total_pol_loss / n,
        total_wdl_loss / n,
        total_moves_left_loss / n,
    )


def _decisive_score(decisive):
    """Promotion-relevant scalar from per-side decisive metrics.

    min() over sides on both components: a checkpoint that collapses for one
    color must not be rescued by the other (WDL-v18 lesson — aggregate
    validation loss picked a checkpoint whose policy top-1 had regressed six
    points and whose Black play collapsed).
    """
    sides = ("white", "black")
    top1 = [decisive[f"policy_top1_{s}"] for s in sides]
    sign = [decisive[f"sign_acc_{s}"] for s in sides]
    if any(v is None for v in top1 + sign):
        return None
    return min(top1) + min(sign)


def _relative_decisive_score(decisive, baseline):
    """Score the worst per-color gains over a fixed incumbent baseline."""
    keys = (
        "policy_top1_white", "policy_top1_black",
        "sign_acc_white", "sign_acc_black",
    )
    if baseline is None or any(
            decisive.get(key) is None or baseline.get(key) is None
            for key in keys):
        return None, {}
    deltas = {key: float(decisive[key] - baseline[key]) for key in keys}
    score = min(deltas["policy_top1_white"],
                deltas["policy_top1_black"])
    score += min(deltas["sign_acc_white"], deltas["sign_acc_black"])
    return float(score), deltas


def _checkpoint_regression_guard(policy_ce, decisive, incumbent_metrics,
                                 max_policy_ce_regression=None,
                                 max_side_top1_drop=None):
    """Protect a stronger selection scalar from material policy regression.

    The reference is either the fixed resume checkpoint or, for legacy runs,
    the currently saved checkpoint. This keeps the rule Pareto-like and
    prevents a tiny sign-score gain from overwriting a much cleaner policy.
    """
    if incumbent_metrics is None:
        return True, []
    reasons = []
    if max_policy_ce_regression is not None:
        limit = incumbent_metrics["policy_ce"] * (1 + max_policy_ce_regression)
        if policy_ce > limit:
            reasons.append(
                f"policy_ce {policy_ce:.4f} exceeds guarded limit {limit:.4f}")
    if max_side_top1_drop is not None:
        for side in ("white", "black"):
            key = f"policy_top1_{side}"
            current = decisive.get(key)
            incumbent = incumbent_metrics.get(key)
            if (current is not None and incumbent is not None
                    and current < incumbent - max_side_top1_drop):
                reasons.append(
                    f"{key} {current:.4f} falls more than "
                    f"{max_side_top1_drop:.4f} below {incumbent:.4f}")
    return not reasons, reasons


@torch.no_grad()
def _eval_epoch(model, loader, device, policy_weight, use_wdl_head=False,
                wdl_loss_weight=0.0, value_head_mode="scalar",
                use_moves_left_head=False, moves_left_loss_weight=0.0,
                use_legal_policy_mask=False):
    model.eval()
    total_loss = 0.0
    total_val_loss = 0.0
    total_pol_loss = 0.0
    total_wdl_loss = 0.0
    total_moves_left_loss = 0.0
    total_wdl_correct = 0.0
    total_wdl_count = 0
    total_mae = 0.0
    total_mse = 0.0
    n = 0
    # Decisive metrics: [correct, count] per (metric, side-to-move).
    dec = {key: [0, 0] for key in (
        "top1_white", "top1_black", "sign_white", "sign_black")}

    for batch in loader:
        (X_b, yv_b, yp_b, ypw_b, yvw_b, yw_b,
         yml_b, ymlw_b, legal_b) = _unpack_aux_loader_batch(
            batch, use_moves_left_head=use_moves_left_head,
            use_wdl_head=use_wdl_head,
            use_legal_policy_mask=use_legal_policy_mask)
        X_b = X_b.to(device)
        yv_b = yv_b.to(device)
        yp_b = yp_b.to(device)
        ypw_b = ypw_b.to(device)
        yvw_b = yvw_b.to(device)
        if yw_b is not None:
            yw_b = yw_b.to(device)
        if yml_b is not None:
            yml_b = yml_b.to(device)
            ymlw_b = ymlw_b.to(device)
        if legal_b is not None:
            legal_b = legal_b.to(device)

        if use_moves_left_head:
            value_pred, policy_pred, wdl_logits, moves_left_pred = (
                model.forward_with_aux(X_b))
        elif use_wdl_head:
            value_pred, policy_pred, wdl_logits = model.forward_with_wdl(X_b)
            moves_left_pred = None
        else:
            value_pred, policy_pred = model(X_b)
            wdl_logits = None
            moves_left_pred = None
        value_loss_pred = value_pred
        loss_val = _power_loss(value_loss_pred, yv_b, weights=yvw_b)
        loss_pol = weighted_policy_cross_entropy(
            policy_pred, yp_b, ypw_b, legal_masks_packed=legal_b)
        loss = loss_val + policy_weight * loss_pol
        loss_wdl = torch.zeros((), device=device)
        if use_wdl_head and wdl_logits is not None and yw_b is not None:
            loss_wdl = _weighted_wdl_ce(wdl_logits, yw_b, yvw_b)
            if wdl_loss_weight > 0:
                loss = loss + wdl_loss_weight * loss_wdl
            preds = torch.argmax(wdl_logits, dim=1)
            total_wdl_correct += (preds == yw_b).sum().item()
            total_wdl_count += int(yw_b.numel())
        loss_moves_left = torch.zeros((), device=device)
        if use_moves_left_head and moves_left_pred is not None and yml_b is not None:
            loss_moves_left = weighted_moves_left_huber(
                moves_left_pred, yml_b, ymlw_b)
            if moves_left_loss_weight > 0:
                loss = loss + moves_left_loss_weight * loss_moves_left

        white_turn = X_b[:, TURN_LAYER, 0, 0] > 0
        pol_enabled = (ypw_b.reshape(-1) > 0) & (yp_b.sum(dim=1) > 0)
        policy_for_metrics = policy_pred
        if legal_b is not None:
            policy_for_metrics, _ = mask_policy_logits(policy_pred, legal_b)
        else:
            policy_for_metrics = mask_inactive_promotion_logits(
                policy_for_metrics, yp_b)
        pol_correct = policy_for_metrics.argmax(dim=1) == yp_b.argmax(dim=1)
        val_nondraw = yv_b.reshape(-1) != 0
        sign_correct = torch.sign(value_pred.reshape(-1)) == torch.sign(yv_b.reshape(-1))
        for side_name, side_mask in (("white", white_turn), ("black", ~white_turn)):
            pm = pol_enabled & side_mask
            dec[f"top1_{side_name}"][0] += int((pol_correct & pm).sum())
            dec[f"top1_{side_name}"][1] += int(pm.sum())
            vm = val_nondraw & side_mask
            dec[f"sign_{side_name}"][0] += int((sign_correct & vm).sum())
            dec[f"sign_{side_name}"][1] += int(vm.sum())

        bs = X_b.size(0)
        total_loss += loss.item() * bs
        total_val_loss += loss_val.item() * bs
        total_pol_loss += loss_pol.item() * bs
        total_wdl_loss += loss_wdl.item() * bs
        total_moves_left_loss += loss_moves_left.item() * bs
        total_mae += (value_pred - yv_b).abs().sum().item()
        total_mse += F.mse_loss(value_pred, yv_b, reduction="sum").item()
        n += bs

    wdl_acc = None
    if total_wdl_count > 0:
        wdl_acc = total_wdl_correct / total_wdl_count
    decisive = {}
    for metric, out_name in (("top1", "policy_top1"), ("sign", "sign_acc")):
        for side in ("white", "black"):
            correct, count = dec[f"{metric}_{side}"]
            decisive[f"{out_name}_{side}"] = correct / count if count else None
    decisive["score"] = _decisive_score(decisive)
    return (
        total_loss / n,
        total_val_loss / n,
        total_pol_loss / n,
        total_mae / n,
        total_mse / n,
        total_wdl_loss / n,
        wdl_acc,
        total_moves_left_loss / n,
        decisive,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Monster Chess dual-head network")
    parser.add_argument("--data-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience in non-improving epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr-gamma", type=float, default=LR_GAMMA,
                        help=f"Per-epoch StepLR decay after warmup "
                             f"(default: {LR_GAMMA})")
    parser.add_argument("--policy-loss-weight", type=float, default=POLICY_LOSS_WEIGHT,
                        help=f"Weight for policy CE term relative to value loss (default: {POLICY_LOSS_WEIGHT})")
    parser.add_argument("--black-policy-weight", type=float, default=1.0,
                        help="Relative policy-example weight for Black-to-move "
                             "positions; weighted CE remains normalized (default: 1.0)")
    parser.add_argument("--train-policy-head-only", action="store_true",
                        help="Freeze the backbone/value head and refine only the "
                             "attention policy head; requires --resume-from")
    parser.add_argument(
        "--train-promotion-head-only", action="store_true",
        help="Freeze v20 and train only distinct-promotion deltas; requires "
             "--promotion-policy and --resume-from")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay for validation/checkpoints; 0 disables")
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--warmup-start-factor", type=float, default=WARMUP_START_FACTOR)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--use-se-blocks", action=argparse.BooleanOptionalAction, default=USE_SE_BLOCKS,
                        help=f"Enable SE modules in residual blocks (default: {USE_SE_BLOCKS})")
    parser.add_argument("--se-reduction", type=int, default=SE_REDUCTION,
                        help=f"SE channel reduction ratio (default: {SE_REDUCTION})")
    parser.add_argument("--target", type=str, default=VALUE_TARGET,
                        choices=["game_result", "mcts_value", "capture_result"],
                        help=f"Value training target (default: {VALUE_TARGET})")
    parser.add_argument("--value-head", type=str, default=VALUE_HEAD_MODE,
                        choices=["scalar", "wdl"],
                        help=f"Value head mode (default: {VALUE_HEAD_MODE})")
    parser.add_argument("--aux-wdl-head", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Train a WDL auxiliary head while retaining the scalar "
                             "value head as the engine output")
    parser.add_argument("--wdl-target", choices=["same", "capture_result"],
                        default="same",
                        help="Source for WDL labels: the primary value target or raw "
                             "capture-only terminal outcomes")
    parser.add_argument("--wdl-loss-weight", type=float, default=WDL_LOSS_WEIGHT,
                        help=f"CE weight for the WDL head (default: {WDL_LOSS_WEIGHT})")
    parser.add_argument("--spatial-value-head", action="store_true",
                        default=SPATIAL_VALUE_HEAD,
                        help="Value head keeps the 8x8 layout (conv 1x1 -> "
                             "flatten -> FC) instead of global average pooling. "
                             "Adds ~529K params; A/B candidate, off by default")
    parser.add_argument("--moves-left-head",
                        action=argparse.BooleanOptionalAction,
                        default=USE_MOVES_LEFT_HEAD,
                        help="Enable the optional remaining-decisions auxiliary head")
    parser.add_argument("--moves-left-loss-weight", type=float,
                        default=MOVES_LEFT_LOSS_WEIGHT,
                        help="Huber-loss weight for --moves-left-head")
    parser.add_argument("--moves-left-head-channels", type=int,
                        default=MOVES_LEFT_HEAD_CHANNELS,
                        help="Hidden width of the moves-left head")
    parser.add_argument("--legal-policy-mask",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Mask illegal logits during policy loss and metrics; "
                             "requires legal_masks_packed.npy")
    parser.add_argument("--wdl-draw-epsilon", type=float, default=WDL_DRAW_EPSILON,
                        help=f"Draw band for WDL labels, |target|<=eps (default: {WDL_DRAW_EPSILON})")
    parser.add_argument("--select-metric", type=str, default="decisive",
                        choices=["decisive", "val_loss"],
                        help="Checkpoint selection: 'decisive' = min-over-sides "
                             "policy top-1 + winner-sign on val (default); "
                             "'val_loss' = legacy aggregate validation loss")
    parser.add_argument(
        "--select-relative-to-resume", action="store_true",
        help="rank epochs by worst-color policy/sign gains over the fixed "
             "--resume-from checkpoint on the same validation split")
    parser.add_argument(
        "--max-policy-ce-regression", type=float, default=None,
        help="Reject a nominally better checkpoint if policy CE exceeds the "
             "saved checkpoint by this relative fraction (for example 0.02)")
    parser.add_argument(
        "--max-side-top1-drop", type=float, default=None,
        help="Reject a nominally better checkpoint if either color's policy "
             "top-1 falls by more than this absolute fraction")
    parser.add_argument(
        "--save-selection-snapshots", action="store_true",
        help="Also preserve every accepted best checkpoint by epoch")
    parser.add_argument("--stem-channels", type=int, default=STEM_CHANNELS,
                        help=f"Stem width (default: {STEM_CHANNELS})")
    parser.add_argument("--policy-head", choices=("dense", "attention"),
                        default=POLICY_HEAD_TYPE,
                        help="Policy head geometry (attention is compact and opt-in)")
    parser.add_argument("--policy-attention-channels", type=int,
                        default=POLICY_ATTENTION_CHANNELS,
                        help="Query/key width for --policy-head=attention")
    parser.add_argument("--side-policy-adapters",
                        action=argparse.BooleanOptionalAction,
                        default=SIDE_POLICY_ADAPTERS,
                        help="Add a small side-conditioned residual projection "
                             "to the attention policy head")
    parser.add_argument(
        "--promotion-policy", action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the 4288-logit policy with distinct q/r/b/n promotions; "
             "requires a promotion-aware processed corpus")
    parser.add_argument("--res-channels", type=str, default=None,
                        help="Comma-separated residual block widths, e.g. "
                             "'96,96,128,128' (default: config tower)")
    args = parser.parse_args()

    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if not (0.0 < args.warmup_start_factor <= 1.0):
        raise ValueError("--warmup-start-factor must be in (0, 1]")
    if args.grad_clip < 0:
        raise ValueError("--grad-clip must be >= 0")
    if args.policy_loss_weight <= 0:
        raise ValueError("--policy-loss-weight must be > 0")
    if args.black_policy_weight <= 0:
        raise ValueError("--black-policy-weight must be > 0")
    if (args.max_policy_ce_regression is not None
            and args.max_policy_ce_regression < 0):
        raise ValueError("--max-policy-ce-regression must be >= 0")
    if args.max_side_top1_drop is not None and args.max_side_top1_drop < 0:
        raise ValueError("--max-side-top1-drop must be >= 0")
    if args.select_relative_to_resume and not args.resume_from:
        raise ValueError("--select-relative-to-resume requires --resume-from")
    if args.train_policy_head_only and not args.resume_from:
        raise ValueError("--train-policy-head-only requires --resume-from")
    if args.train_promotion_head_only and not args.resume_from:
        raise ValueError("--train-promotion-head-only requires --resume-from")
    if args.train_promotion_head_only and not args.promotion_policy:
        raise ValueError("--train-promotion-head-only requires --promotion-policy")
    if args.train_promotion_head_only and args.train_policy_head_only:
        raise ValueError("head-only training modes are mutually exclusive")
    if not (0.0 < args.lr_gamma <= 1.0):
        raise ValueError("--lr-gamma must be in (0, 1]")
    if args.se_reduction <= 0:
        raise ValueError("--se-reduction must be > 0")
    if args.wdl_loss_weight < 0:
        raise ValueError("--wdl-loss-weight must be >= 0")
    if args.wdl_draw_epsilon < 0:
        raise ValueError("--wdl-draw-epsilon must be >= 0")
    if args.moves_left_loss_weight < 0:
        raise ValueError("--moves-left-loss-weight must be >= 0")
    if args.moves_left_head_channels <= 0:
        raise ValueError("--moves-left-head-channels must be > 0")
    if args.policy_attention_channels <= 0:
        raise ValueError("--policy-attention-channels must be > 0")
    if args.side_policy_adapters and args.policy_head != "attention":
        raise ValueError("--side-policy-adapters requires --policy-head=attention")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        raise ValueError("--ema-decay must be 0 (off) or in (0, 1)")
    if args.aux_wdl_head and args.value_head == "wdl":
        raise ValueError("--aux-wdl-head is only valid with --value-head=scalar")
    use_wdl_mode = args.value_head == "wdl" or args.aux_wdl_head
    if not use_wdl_mode and args.wdl_loss_weight > 0:
        print("Warning: --wdl-loss-weight ignored because --value-head=scalar")
    if not use_wdl_mode and args.wdl_target != "same":
        print("Warning: --wdl-target ignored because no WDL head is enabled")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"Loading data from {args.data_dir}...")
    loaded = load_data(
        args.data_dir, include_moves_left=args.moves_left_head,
        include_legal_masks=args.legal_policy_mask,
        include_capture_results=(args.target == "capture_result" or (
            use_wdl_mode and args.wdl_target == "capture_result")))
    (positions, mcts_values, game_results, policies, policy_weights,
     value_weights, splits) = loaded[:7]
    expected_policy_size = (PROMOTION_AWARE_POLICY_SIZE
                            if args.promotion_policy else POLICY_SIZE)
    if policies.ndim != 2 or policies.shape[1] != expected_policy_size:
        raise ValueError(
            f"--promotion-policy={args.promotion_policy} expects policies.npy "
            f"width {expected_policy_size}, got {policies.shape}")
    policy_weights = apply_black_policy_weight(
        policy_weights, positions, args.black_policy_weight)
    cursor = 7
    moves_left = moves_left_weights = None
    if args.moves_left_head:
        moves_left, moves_left_weights = loaded[cursor:cursor + 2]
        cursor += 2
    legal_masks_packed = None
    if args.legal_policy_mask:
        legal_masks_packed = loaded[cursor]
        cursor += 1
    capture_results = None
    if args.target == "capture_result" or (
            use_wdl_mode and args.wdl_target == "capture_result"):
        capture_results = loaded[cursor]

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Empty split detected (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}). "
            "Regenerate processed data with enough games per split."
        )

    game_results_side = to_side_perspective(game_results, positions)
    capture_results_side = (
        to_side_perspective(capture_results, positions)
        if capture_results is not None else None)
    value_results_side = (
        capture_results_side
        if args.target == "capture_result" else game_results_side)
    wdl_results_side = (
        capture_results_side
        if args.wdl_target == "capture_result" else value_results_side)
    wdl_targets = build_wdl_targets(
        wdl_results_side, draw_epsilon=args.wdl_draw_epsilon)
    value_targets = get_targets(
        mcts_values, value_results_side, args.target)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Value target: {args.target}")
    if use_wdl_mode:
        losses = int((wdl_targets == 0).sum())
        draws = int((wdl_targets == 1).sum())
        wins = int((wdl_targets == 2).sum())
        print(
            "WDL targets: "
            f"loss={losses} draw={draws} win={wins} "
            f"(draw_epsilon={args.wdl_draw_epsilon:.3f}, ce_w={args.wdl_loss_weight:.3f})"
        )
        print(f"WDL target source: {args.wdl_target}")
    print(f"Policy loss weight: {args.policy_loss_weight}")
    print(f"Black policy-example weight: {args.black_policy_weight}")
    if args.moves_left_head:
        enabled = int((moves_left_weights > 0).sum())
        print(f"Moves-left head: enabled (Huber weight="
              f"{args.moves_left_loss_weight}, trusted={enabled}/"
              f"{len(moves_left_weights)})")
    print(f"Legal policy masking: {args.legal_policy_mask}")

    # Build model. Input channels come from the processed data, not config:
    # a 15-plane corpus trains a 15-plane model even when config default is 17.
    data_channels = int(positions.shape[3])
    res_channels = (tuple(int(c) for c in args.res_channels.split(","))
                    if args.res_channels else RESIDUAL_BLOCK_CHANNELS)
    model = build_model(
        input_channels=data_channels,
        policy_head_type=args.policy_head,
        policy_attention_channels=args.policy_attention_channels,
        side_policy_adapters=args.side_policy_adapters,
        promotion_policy=args.promotion_policy,
        stem_channels=args.stem_channels,
        residual_block_channels=res_channels,
        use_se_blocks=args.use_se_blocks,
        se_reduction=args.se_reduction,
        use_wdl_head=use_wdl_mode,
        value_head_mode=args.value_head,
        spatial_value_head=args.spatial_value_head,
        use_moves_left_head=args.moves_left_head,
        moves_left_head_channels=args.moves_left_head_channels,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Value head: {'spatial (8x8 preserved)' if model.spatial_value_head else 'GAP (channel means)'}")
    print(f"Input channels: {model.input_channels}")
    print(f"Policy head channels: {model.policy_head_channels}")
    print(f"Policy head: {model.policy_head_type} "
          f"(attention channels={model.policy_attention_channels})")
    print(f"Side policy adapters: {model.side_policy_adapters}")
    print(f"Promotion-aware policy: {model.promotion_policy} "
          f"(output width={model.policy_output_size})")
    print(f"Stem channels: {model.stem_channels}")
    print(f"Residual blocks: {list(model.residual_block_channels)}")
    print(f"SE blocks: {model.use_se_blocks} (reduction={model.se_reduction})")
    print(f"Value head mode: {model.value_head_mode} (wdl_head={model.use_wdl_head})")
    print(f"Moves-left head: {model.use_moves_left_head} "
          f"(channels={model.moves_left_head_channels})")
    resume_loaded_count = None
    resume_skipped = None
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume-from not found: {args.resume_from}")
        resume_state = torch.load(args.resume_from, map_location=device, weights_only=True)
        loaded_count, skipped_keys = load_state_dict_flexible(model, resume_state)
        resume_loaded_count = loaded_count
        resume_skipped = skipped_keys
        print(
            f"Resumed weights from {args.resume_from}: "
            f"loaded {loaded_count} tensors, skipped {len(skipped_keys)} incompatible"
        )

    if args.train_promotion_head_only:
        trainable_parameter_names = configure_promotion_head_only(model, True)
    else:
        trainable_parameter_names = configure_policy_head_only(
            model, args.train_policy_head_only)
    if args.train_policy_head_only:
        print("Training policy head only: " + ", ".join(trainable_parameter_names))
    if args.train_promotion_head_only:
        print("Training promotion head only: "
              + ", ".join(trainable_parameter_names))

    optimizer, decay_count, no_decay_count = build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
    )
    print(f"Optimizer: AdamW (weight_decay={args.weight_decay})")
    print(f"  Param groups: decay={decay_count}, no_decay={no_decay_count}")
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    print(f"EMA: {'off' if ema is None else f'decay={args.ema_decay}'}")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=args.lr_gamma,
    )

    # Checkpoint setup
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, "best_value_net.pt")
    best_selection_value = float("inf")
    best_checkpoint_metrics = None
    best_epoch = None
    patience_counter = 0
    patience = args.patience
    run_id = time.strftime("%Y%m%d_%H%M%S")
    metadata_path = os.path.join(args.model_dir, f"train_run_{run_id}.json")
    run_metadata = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": get_git_commit(),
        "seed": args.seed,
        "device": str(device),
        "args": vars(args),
        "model_params": total_params,
        "input_channels": int(model.input_channels),
        "policy_head_channels": int(model.policy_head_channels),
        "policy_head_type": str(model.policy_head_type),
        "policy_attention_channels": int(model.policy_attention_channels),
        "side_policy_adapters": bool(model.side_policy_adapters),
        "promotion_policy": bool(model.promotion_policy),
        "policy_output_size": int(model.policy_output_size),
        "stem_channels": int(model.stem_channels),
        "residual_block_channels": [int(x) for x in model.residual_block_channels],
        "residual_block_count": int(model.residual_block_count),
        "use_se_blocks": bool(model.use_se_blocks),
        "se_reduction": int(model.se_reduction),
        "value_head_mode": str(model.value_head_mode),
        "use_wdl_head": bool(model.use_wdl_head),
        "use_moves_left_head": bool(model.use_moves_left_head),
        "legal_policy_mask": bool(args.legal_policy_mask),
        "moves_left_head_channels": int(model.moves_left_head_channels),
        "wdl_draw_epsilon": float(args.wdl_draw_epsilon),
        "wdl_loss_weight": float(args.wdl_loss_weight),
        "resume_from": args.resume_from,
        "resume_loaded_tensors": int(resume_loaded_count) if resume_loaded_count is not None else None,
        "resume_skipped_keys": resume_skipped if resume_skipped is not None else [],
        "train_policy_head_only": bool(args.train_policy_head_only),
        "train_promotion_head_only": bool(args.train_promotion_head_only),
        "trainable_parameter_names": trainable_parameter_names,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": args.weight_decay,
            "decay_group_count": decay_count,
            "no_decay_group_count": no_decay_count,
        },
        "ema": {
            "enabled": ema is not None,
            "decay": float(args.ema_decay),
            "validation_and_checkpoint_use_ema": ema is not None,
        },
        "loss_weights": {
            "policy": float(args.policy_loss_weight),
            "value": 1.0,
            "moves_left": (float(args.moves_left_loss_weight)
                            if args.moves_left_head else 0.0),
        },
        "warmup": {
            "epochs": args.warmup_epochs,
            "start_factor": args.warmup_start_factor,
        },
        "scheduler": {
            "name": "StepLR",
            "gamma": float(args.lr_gamma),
            "after_warmup": True,
        },
        "gradient_clipping": {
            "max_norm": args.grad_clip,
        },
        "data_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "epochs": [],
    }

    selection_baseline = None
    if args.select_relative_to_resume:
        baseline_loader = _make_loader(
            positions[val_idx], value_targets[val_idx], policies[val_idx],
            args.batch_size, shuffle=False,
            y_wdl=wdl_targets[val_idx] if use_wdl_mode else None,
            y_policy_weight=policy_weights[val_idx],
            y_value_weight=value_weights[val_idx],
            y_moves_left=(moves_left[val_idx]
                          if args.moves_left_head else None),
            y_moves_left_weight=(moves_left_weights[val_idx]
                                 if args.moves_left_head else None),
            y_legal_masks_packed=(legal_masks_packed[val_idx]
                                  if args.legal_policy_mask else None),
        )
        baseline_model = ema.module if ema is not None else model
        (_base_loss, _base_v, base_policy_ce, _base_mae, _base_mse,
         _base_wdl, _base_wdl_acc, _base_moves_left,
         base_decisive) = _eval_epoch(
            baseline_model, baseline_loader, device, args.policy_loss_weight,
            use_wdl_head=use_wdl_mode,
            wdl_loss_weight=(args.wdl_loss_weight if use_wdl_mode else 0.0),
            value_head_mode=args.value_head,
            use_moves_left_head=args.moves_left_head,
            moves_left_loss_weight=(args.moves_left_loss_weight
                                    if args.moves_left_head else 0.0),
            use_legal_policy_mask=args.legal_policy_mask,
        )
        selection_baseline = {
            "policy_ce": float(base_policy_ce),
            **{key: (float(value) if value is not None else None)
               for key, value in base_decisive.items()},
        }
        run_metadata["selection_baseline"] = selection_baseline
        def _percent(value):
            return "n/a" if value is None else f"{value:.1%}"
        print(
            "Selection baseline: "
            f"policy_ce={base_policy_ce:.4f} "
            f"top1(W={_percent(base_decisive['policy_top1_white'])} "
            f"B={_percent(base_decisive['policy_top1_black'])}) "
            f"sign(W={_percent(base_decisive['sign_acc_white'])} "
            f"B={_percent(base_decisive['sign_acc_black'])})"
        )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_targets = value_targets

        lr_used = _set_epoch_lr(
            optimizer,
            epoch=epoch,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            warmup_start_factor=args.warmup_start_factor,
        )

        train_gen = torch.Generator()
        train_gen.manual_seed(args.seed + epoch)
        epoch_train_idx = train_idx
        val_loader = _make_loader(
            positions[val_idx],
            epoch_targets[val_idx],
            policies[val_idx],
            args.batch_size,
            shuffle=False,
            y_wdl=wdl_targets[val_idx] if use_wdl_mode else None,
            y_policy_weight=policy_weights[val_idx],
            y_value_weight=value_weights[val_idx],
            y_moves_left=(moves_left[val_idx]
                          if args.moves_left_head else None),
            y_moves_left_weight=(moves_left_weights[val_idx]
                                 if args.moves_left_head else None),
            y_legal_masks_packed=(legal_masks_packed[val_idx]
                                  if args.legal_policy_mask else None),
        )

        # Keep policy labels aligned with selected train indices.
        train_loader = _make_loader(
            positions[epoch_train_idx],
            epoch_targets[epoch_train_idx],
            policies[epoch_train_idx],
            args.batch_size,
            shuffle=True,
            generator=train_gen,
            y_wdl=wdl_targets[epoch_train_idx] if use_wdl_mode else None,
            y_policy_weight=policy_weights[epoch_train_idx],
            y_value_weight=value_weights[epoch_train_idx],
            y_moves_left=(moves_left[epoch_train_idx]
                          if args.moves_left_head else None),
            y_moves_left_weight=(moves_left_weights[epoch_train_idx]
                                 if args.moves_left_head else None),
            y_legal_masks_packed=(legal_masks_packed[epoch_train_idx]
                                  if args.legal_policy_mask else None),
        )

        train_loss, train_v, train_p, train_wdl, train_moves_left = _train_epoch(
            model, train_loader, optimizer, device, args.policy_loss_weight, args.grad_clip,
            use_wdl_head=use_wdl_mode,
            wdl_loss_weight=args.wdl_loss_weight if use_wdl_mode else 0.0,
            value_head_mode=args.value_head,
            use_moves_left_head=args.moves_left_head,
            moves_left_loss_weight=(args.moves_left_loss_weight
                                    if args.moves_left_head else 0.0),
            use_legal_policy_mask=args.legal_policy_mask,
            ema=ema,
        )
        eval_model = ema.module if ema is not None else model
        (val_loss, val_v, val_p, val_mae, val_mse, val_wdl, val_wdl_acc,
         val_moves_left, val_decisive) = _eval_epoch(
            eval_model, val_loader, device, args.policy_loss_weight,
            use_wdl_head=use_wdl_mode,
            wdl_loss_weight=args.wdl_loss_weight if use_wdl_mode else 0.0,
            value_head_mode=args.value_head,
            use_moves_left_head=args.moves_left_head,
            moves_left_loss_weight=(args.moves_left_loss_weight
                                    if args.moves_left_head else 0.0),
            use_legal_policy_mask=args.legal_policy_mask,
        )
        if epoch > args.warmup_epochs:
            scheduler.step()

        lr = lr_used
        wdl_str = ""
        if use_wdl_mode:
            acc_str = f"{val_wdl_acc:.1%}" if val_wdl_acc is not None else "n/a"
            wdl_str = f"  wdl(train_ce={train_wdl:.4f} val_ce={val_wdl:.4f} val_acc={acc_str})"
        ml_str = ""
        if args.moves_left_head:
            ml_str = (f"  moves_left(train_huber={train_moves_left:.4f} "
                      f"val_huber={val_moves_left:.4f})")
        dec_str = "  decisive(n/a)"
        if val_decisive["score"] is not None:
            dec_str = (
                f"  decisive(top1 W={val_decisive['policy_top1_white']:.1%} "
                f"B={val_decisive['policy_top1_black']:.1%} "
                f"sign W={val_decisive['sign_acc_white']:.1%} "
                f"B={val_decisive['sign_acc_black']:.1%})"
            )
        print(f"Epoch {epoch:3d}  "
              f"train={train_loss:.4f} (v={train_v:.4f} p={train_p:.4f})  "
              f"val={val_loss:.4f} (pow={val_v:.4f} mse={val_mse:.4f} p={val_p:.4f} mae={val_mae:.4f})  "
              f"lr={lr:.1e}{wdl_str}{ml_str}{dec_str}")
        run_metadata["epochs"].append({
            "epoch": epoch,
            "train_samples": int(len(epoch_train_idx)),
            "train_total_loss": float(train_loss),
            "train_value_power_loss": float(train_v),
            "train_policy_ce": float(train_p),
            "train_wdl_ce": float(train_wdl),
            "train_moves_left_huber": float(train_moves_left),
            "val_total_loss": float(val_loss),
            "val_value_power_loss": float(val_v),
            "val_value_mse": float(val_mse),
            "val_policy_ce": float(val_p),
            "val_value_mae": float(val_mae),
            "val_wdl_ce": float(val_wdl),
            "val_wdl_accuracy": float(val_wdl_acc) if val_wdl_acc is not None else None,
            "val_moves_left_huber": float(val_moves_left),
            "val_decisive": {k: (float(v) if v is not None else None)
                             for k, v in val_decisive.items()},
            "lr": float(lr),
        })

        # Checkpoint selection. "decisive" maximizes min-over-sides policy
        # top-1 + winner-sign; falls back to val_loss only if a side has no
        # val samples. Lower selection value = better for both modes.
        relative_score = None
        relative_deltas = {}
        if args.select_relative_to_resume:
            relative_score, relative_deltas = _relative_decisive_score(
                val_decisive, selection_baseline)
        if relative_score is not None:
            selection_value = -relative_score
            selection_desc = f"incumbent_relative_score={relative_score:+.4f}"
        elif args.select_metric == "decisive" and val_decisive["score"] is not None:
            selection_value = -val_decisive["score"]
            selection_desc = f"decisive_score={val_decisive['score']:.4f}"
        else:
            selection_value = val_loss
            selection_desc = f"val_loss={val_loss:.4f}"
        guard_ok, guard_reasons = _checkpoint_regression_guard(
            val_p, val_decisive,
            selection_baseline if args.select_relative_to_resume
            else best_checkpoint_metrics,
            max_policy_ce_regression=args.max_policy_ce_regression,
            max_side_top1_drop=args.max_side_top1_drop,
        )
        nominal_improvement = selection_value < best_selection_value
        should_stop = False
        if nominal_improvement and guard_ok:
            best_selection_value = selection_value
            best_epoch = epoch
            best_checkpoint_metrics = {
                "policy_ce": float(val_p),
                "policy_top1_white": val_decisive["policy_top1_white"],
                "policy_top1_black": val_decisive["policy_top1_black"],
            }
            patience_counter = 0
            torch.save(eval_model.state_dict(), checkpoint_path)
            if args.save_selection_snapshots:
                torch.save(
                    eval_model.state_dict(),
                    os.path.join(args.model_dir,
                                 f"selected_epoch_{epoch:03d}.pt"),
                )
            print(f"  -> saved best model ({selection_desc})")
        else:
            if nominal_improvement and not guard_ok:
                print("  -> checkpoint rejected by regression guard: "
                      + "; ".join(guard_reasons))
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                should_stop = True
        run_metadata["epochs"][-1]["checkpoint"] = {
            "nominal_improvement": bool(nominal_improvement),
            "guard_passed": bool(guard_ok),
            "guard_reasons": guard_reasons,
            "saved": bool(nominal_improvement and guard_ok),
            "relative_score": relative_score,
            "relative_deltas": relative_deltas,
        }
        if should_stop:
            break

    if best_epoch is None:
        rejection = {
            "status": "rejected_training",
            "reason": "no epoch passed fixed incumbent regression guards",
            "selection_baseline": selection_baseline,
            "epochs": run_metadata["epochs"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rejection_path = os.path.join(args.model_dir, "selection_rejected.json")
        run_metadata["status"] = "rejected_training"
        run_metadata["best_selection_value"] = None
        run_metadata["best_epoch"] = None
        run_metadata["checkpoint_path"] = None
        run_metadata["end_timestamp"] = rejection["timestamp"]
        with open(metadata_path, "w") as f:
            json.dump(run_metadata, f, indent=2)
        with open(rejection_path, "w") as f:
            json.dump(rejection, f, indent=2)
        print("TRAINING SELECTION: REJECT — no epoch passed the fixed "
              "incumbent guards")
        print(f"Selection rejection saved to {rejection_path}")
        raise SystemExit(2)

    # Load best model for test evaluation
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_loader = _make_loader(
        positions[test_idx],
        value_targets[test_idx],
        policies[test_idx],
        args.batch_size,
        shuffle=False,
        y_wdl=wdl_targets[test_idx] if use_wdl_mode else None,
        y_policy_weight=policy_weights[test_idx],
        y_value_weight=value_weights[test_idx],
        y_moves_left=(moves_left[test_idx] if args.moves_left_head else None),
        y_moves_left_weight=(moves_left_weights[test_idx]
                             if args.moves_left_head else None),
        y_legal_masks_packed=(legal_masks_packed[test_idx]
                              if args.legal_policy_mask else None),
    )
    (test_loss, test_v, test_p, test_mae, test_mse, test_wdl, test_wdl_acc,
     test_moves_left, test_decisive) = _eval_epoch(
        model, test_loader, device, args.policy_loss_weight,
        use_wdl_head=use_wdl_mode,
        wdl_loss_weight=args.wdl_loss_weight if use_wdl_mode else 0.0,
        value_head_mode=args.value_head,
        use_moves_left_head=args.moves_left_head,
        moves_left_loss_weight=(args.moves_left_loss_weight
                                if args.moves_left_head else 0.0),
        use_legal_policy_mask=args.legal_policy_mask,
    )

    print("\n--- Test set evaluation ---")
    print(f"Total loss: {test_loss:.4f}")
    print(f"Value power loss: {test_v:.4f}")
    print(f"Value true MSE:   {test_mse:.4f}")
    print(f"Policy CE:  {test_p:.4f}")
    print(f"Value MAE:  {test_mae:.4f}")
    if use_wdl_mode:
        acc_str = f"{test_wdl_acc:.1%}" if test_wdl_acc is not None else "n/a"
        print(f"WDL CE:     {test_wdl:.4f}")
        print(f"WDL Acc:    {acc_str}")
    if args.moves_left_head:
        print(f"Moves-left Huber: {test_moves_left:.4f}")
    if test_decisive["score"] is not None:
        print(f"Policy top-1 (enabled): W={test_decisive['policy_top1_white']:.1%} "
              f"B={test_decisive['policy_top1_black']:.1%}")
        print(f"Winner sign (non-draw): W={test_decisive['sign_acc_white']:.1%} "
              f"B={test_decisive['sign_acc_black']:.1%}")

    # Value sign-accuracy
    model.eval()
    all_vpreds = []
    all_vtrue = []
    with torch.no_grad():
        for batch in test_loader:
            X_b, yv_b, _, _, _, _, _, _, _ = _unpack_aux_loader_batch(
                batch, use_moves_left_head=args.moves_left_head,
                use_wdl_head=use_wdl_mode,
                use_legal_policy_mask=args.legal_policy_mask)
            vp, _ = model(X_b.to(device))
            all_vpreds.append(vp.cpu().numpy())
            all_vtrue.append(yv_b.numpy())
    vpreds = np.concatenate(all_vpreds).flatten()
    vtrue = np.concatenate(all_vtrue).flatten()
    non_draw = vtrue != 0
    if non_draw.sum() > 0:
        acc = np.mean(np.sign(vpreds[non_draw]) == np.sign(vtrue[non_draw]))
        print(f"Winner prediction accuracy (non-draw): {acc:.1%}")

    run_metadata["test"] = {
        "total_loss": float(test_loss),
        "value_power_loss": float(test_v),
        "value_true_mse": float(test_mse),
        "policy_ce": float(test_p),
        "value_mae": float(test_mae),
        "wdl_ce": float(test_wdl),
        "wdl_accuracy": float(test_wdl_acc) if test_wdl_acc is not None else None,
        "moves_left_huber": float(test_moves_left),
        "winner_sign_accuracy_non_draw": float(acc) if non_draw.sum() > 0 else None,
        "decisive": {k: (float(v) if v is not None else None)
                     for k, v in test_decisive.items()},
    }
    run_metadata["select_metric"] = args.select_metric
    run_metadata["best_selection_value"] = float(best_selection_value)
    run_metadata["best_epoch"] = best_epoch
    run_metadata["checkpoint_path"] = checkpoint_path
    run_metadata["end_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"\nBest model saved to {checkpoint_path}")
    print(f"Run metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
