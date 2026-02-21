import argparse
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
    TENSOR_SHAPE, TURN_LAYER, POLICY_SIZE, POLICY_LOSS_WEIGHT,
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    VALUE_TARGET, BLEND_WEIGHT, BLEND_START, BLEND_END,
    VALUE_HEAD_MODE, WDL_LOSS_WEIGHT, WDL_DRAW_EPSILON,
    PROCESSED_DATA_DIR, MODEL_DIR,
    VALUE_LOSS_EXPONENT, LR_GAMMA, RANDOM_SEED,
    WEIGHT_DECAY, GRAD_CLIP_NORM, WARMUP_EPOCHS, WARMUP_START_FACTOR,
    POLICY_HEAD_CHANNELS, STEM_CHANNELS, RESIDUAL_BLOCK_CHANNELS,
    USE_SE_BLOCKS, SE_REDUCTION, USE_SIDE_SPECIALIZED_HEADS,
)

# Input channels = last dim of TENSOR_SHAPE (8, 8, 15)
IN_CHANNELS = TENSOR_SHAPE[2]
SOURCE_ID_HUMAN = 1  # SOURCE_ORDER index in data_processor.py


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
    """Dual-head ResNet: (N, 15, 8, 8) -> (value in [-1,1], policy logits[4096]).

    Shared backbone:
      Configurable stem and residual tower from config.py
    Value head:
      GAP -> Dense 128 -> Dense 64 -> Dense 1 (tanh)
    Policy head:
      Conv Cx1x1 -> BN -> ReLU -> Flatten -> Dense 4096 (logits)
    """
    def __init__(
        self,
        policy_head_channels=POLICY_HEAD_CHANNELS,
        stem_channels=STEM_CHANNELS,
        residual_block_channels=RESIDUAL_BLOCK_CHANNELS,
        use_se_blocks=USE_SE_BLOCKS,
        se_reduction=SE_REDUCTION,
        use_side_specialized_heads=USE_SIDE_SPECIALIZED_HEADS,
        use_wdl_head=False,
        value_head_mode=VALUE_HEAD_MODE,
    ):
        super().__init__()
        self.policy_head_channels = int(policy_head_channels)
        self.stem_channels = int(stem_channels)
        self.residual_block_channels = tuple(int(c) for c in residual_block_channels)
        self.use_se_blocks = bool(use_se_blocks)
        self.se_reduction = int(se_reduction)
        self.use_side_specialized_heads = bool(use_side_specialized_heads)
        self.use_wdl_head = bool(use_wdl_head)
        self.value_head_mode = str(value_head_mode)
        if not self.residual_block_channels:
            raise ValueError("residual_block_channels must contain at least one block")
        if self.se_reduction <= 0:
            raise ValueError("se_reduction must be > 0")
        if self.value_head_mode not in ("scalar", "wdl"):
            raise ValueError("value_head_mode must be 'scalar' or 'wdl'")
        if self.value_head_mode == "wdl" and not self.use_wdl_head:
            raise ValueError("value_head_mode='wdl' requires use_wdl_head=True")

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, self.stem_channels, 3, padding=1, bias=False),
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
        if self.use_side_specialized_heads:
            self.value_head_white = self._make_value_head()
            self.value_head_black = self._make_value_head()
        if self.use_wdl_head:
            self.wdl_head = self._make_wdl_head()
            if self.use_side_specialized_heads:
                self.wdl_head_white = self._make_wdl_head()
                self.wdl_head_black = self._make_wdl_head()

        # Policy head(s)
        self.policy_conv, self.policy_fc = self._make_policy_head()
        if self.use_side_specialized_heads:
            self.policy_conv_white, self.policy_fc_white = self._make_policy_head()
            self.policy_conv_black, self.policy_fc_black = self._make_policy_head()
            # Initialize side-specific heads from the shared heads for smoother warm start.
            self.value_head_white.load_state_dict(self.value_head.state_dict())
            self.value_head_black.load_state_dict(self.value_head.state_dict())
            self.policy_conv_white.load_state_dict(self.policy_conv.state_dict())
            self.policy_conv_black.load_state_dict(self.policy_conv.state_dict())
            self.policy_fc_white.load_state_dict(self.policy_fc.state_dict())
            self.policy_fc_black.load_state_dict(self.policy_fc.state_dict())
            if self.use_wdl_head:
                self.wdl_head_white.load_state_dict(self.wdl_head.state_dict())
                self.wdl_head_black.load_state_dict(self.wdl_head.state_dict())

    def _make_value_head(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (N, C, 1, 1)
            nn.Flatten(),                  # (N, C)
            nn.Linear(self.backbone_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def _make_wdl_head(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),  # [loss, draw, win]
        )

    def _make_policy_head(self):
        conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, self.policy_head_channels, 1, bias=False),
            nn.BatchNorm2d(self.policy_head_channels),
            nn.ReLU(),
        )
        fc = nn.Linear(self.policy_head_channels * 8 * 8, POLICY_SIZE)
        return conv, fc

    @staticmethod
    def _policy_logits(backbone, policy_conv, policy_fc):
        p = policy_conv(backbone)
        p = p.flatten(1)
        return policy_fc(p)

    def _forward_backbone(self, x):
        # x: (N, C, 8, 8)  — PyTorch uses channels-first
        side_turn = x[:, TURN_LAYER, 0, 0]
        x = self.stem(x)
        for i in range(1, self.residual_block_count + 1):
            x = getattr(self, f"res{i}")(x)
        return x, side_turn

    def _compute_heads(self, backbone, side_turn):
        value = self.value_head(backbone)
        policy = self._policy_logits(backbone, self.policy_conv, self.policy_fc)
        wdl_logits = self.wdl_head(backbone) if self.use_wdl_head else None
        if self.use_side_specialized_heads:
            white_mask = (side_turn > 0).unsqueeze(1)
            value_white = self.value_head_white(backbone)
            value_black = self.value_head_black(backbone)
            policy_white = self._policy_logits(backbone, self.policy_conv_white, self.policy_fc_white)
            policy_black = self._policy_logits(backbone, self.policy_conv_black, self.policy_fc_black)
            value = torch.where(white_mask, value_white, value_black)
            policy = torch.where(
                white_mask.expand(-1, POLICY_SIZE),
                policy_white,
                policy_black,
            )
            if self.use_wdl_head:
                wdl_white = self.wdl_head_white(backbone)
                wdl_black = self.wdl_head_black(backbone)
                wdl_logits = torch.where(
                    white_mask.expand(-1, 3),
                    wdl_white,
                    wdl_black,
                )

        if self.use_wdl_head and self.value_head_mode == "wdl":
            value = _wdl_expectation_from_logits(wdl_logits)

        return value, policy, wdl_logits

    def forward(self, x):
        backbone, side_turn = self._forward_backbone(x)
        value, policy, _ = self._compute_heads(backbone, side_turn)
        return value, policy

    def forward_with_wdl(self, x):
        backbone, side_turn = self._forward_backbone(x)
        return self._compute_heads(backbone, side_turn)


def infer_policy_head_channels(state_dict):
    """Infer policy bottleneck width from checkpoint state dict."""
    w = state_dict.get("policy_conv.0.weight")
    if w is None:
        w = state_dict.get("policy_conv_white.0.weight")
    if w is None:
        w = state_dict.get("policy_conv_black.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[0] > 0:
        return int(w.shape[0])
    return POLICY_HEAD_CHANNELS


def infer_side_head_config(state_dict):
    """Infer whether side-specialized heads are present in checkpoint."""
    return (
        "value_head_white.2.weight" in state_dict
        or "policy_conv_white.0.weight" in state_dict
    )


def infer_wdl_head_config(state_dict):
    """Infer whether WDL head exists in checkpoint."""
    has_wdl = any(
        k.startswith("wdl_head.") or k.startswith("wdl_head_white.") or k.startswith("wdl_head_black.")
        for k in state_dict.keys()
    )
    if not has_wdl:
        return False, "scalar"
    # If WDL head tensors exist, treat this checkpoint as WDL mode.
    return True, "wdl"


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
    policy_head_channels=POLICY_HEAD_CHANNELS,
    stem_channels=STEM_CHANNELS,
    residual_block_channels=RESIDUAL_BLOCK_CHANNELS,
    use_se_blocks=USE_SE_BLOCKS,
    se_reduction=SE_REDUCTION,
    use_side_specialized_heads=USE_SIDE_SPECIALIZED_HEADS,
    use_wdl_head=False,
    value_head_mode=VALUE_HEAD_MODE,
):
    return DualHeadNet(
        policy_head_channels=policy_head_channels,
        stem_channels=stem_channels,
        residual_block_channels=residual_block_channels,
        use_se_blocks=use_se_blocks,
        se_reduction=se_reduction,
        use_side_specialized_heads=use_side_specialized_heads,
        use_wdl_head=use_wdl_head,
        value_head_mode=value_head_mode,
    )


def load_model_for_inference(checkpoint_path, device):
    """Load model with architecture inferred from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    pol_ch = infer_policy_head_channels(state_dict)
    stem_ch, block_ch = infer_backbone_architecture(state_dict)
    use_se_blocks, se_reduction = infer_se_config(state_dict)
    use_side_specialized_heads = infer_side_head_config(state_dict)
    use_wdl_head, value_head_mode = infer_wdl_head_config(state_dict)
    model = build_model(
        policy_head_channels=pol_ch,
        stem_channels=stem_ch,
        residual_block_channels=block_ch,
        use_se_blocks=use_se_blocks,
        se_reduction=se_reduction,
        use_side_specialized_heads=use_side_specialized_heads,
        use_wdl_head=use_wdl_head,
        value_head_mode=value_head_mode,
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


def load_data(data_dir):
    """Load processed training data and splits."""
    positions = np.load(os.path.join(data_dir, "positions.npy"))
    mcts_values = np.load(os.path.join(data_dir, "mcts_values.npy"))
    game_results = np.load(os.path.join(data_dir, "game_results.npy"))
    policies = np.load(os.path.join(data_dir, "policies.npy"))
    target_lambdas_path = os.path.join(data_dir, "target_lambdas.npy")
    if os.path.exists(target_lambdas_path):
        target_lambdas = np.load(target_lambdas_path)
    else:
        target_lambdas = np.ones((len(positions),), dtype=np.float32)
    source_ids_path = os.path.join(data_dir, "source_ids.npy")
    source_ids = None
    if os.path.exists(source_ids_path):
        source_ids = np.load(source_ids_path)
    splits = np.load(os.path.join(data_dir, "splits.npz"))
    return positions, mcts_values, game_results, policies, target_lambdas, source_ids, splits


def to_side_perspective(values_white_perspective, positions):
    """Convert white-perspective targets to side-to-move perspective."""
    side_sign = np.where(positions[:, 0, 0, TURN_LAYER] > 0, 1.0, -1.0).astype(np.float32)
    return values_white_perspective * side_sign


def get_targets(mcts_values, game_results, target_type, blend_weight, target_lambdas=None):
    """Build value training targets based on the chosen strategy."""
    if target_type == "game_result":
        return game_results
    elif target_type == "mcts_value":
        return mcts_values
    elif target_type == "blend":
        return blend_weight * mcts_values + (1 - blend_weight) * game_results
    elif target_type == "source_aware":
        if target_lambdas is None:
            raise ValueError("source_aware target requires per-sample target_lambdas")
        return target_lambdas * mcts_values + (1 - target_lambdas) * game_results
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


def get_blend_lambda(epoch, max_epochs):
    """Anneal blend weight from BLEND_START to BLEND_END over training."""
    if max_epochs <= 1:
        return BLEND_START
    t = (epoch - 1) / (max_epochs - 1)
    return BLEND_START + (BLEND_END - BLEND_START) * t


def _make_loader(X, y_val, y_pol, batch_size, shuffle=True, generator=None, y_wdl=None):
    """Create a DataLoader from numpy arrays.

    Transposes X from (N, 8, 8, C) to (N, C, 8, 8) for PyTorch.
    """
    X_t = torch.from_numpy(X.transpose(0, 3, 1, 2))  # channels-first
    y_v = torch.from_numpy(y_val).unsqueeze(1)         # (N, 1)
    y_p = torch.from_numpy(y_pol)                      # (N, 4096)
    if y_wdl is None:
        ds = TensorDataset(X_t, y_v, y_p)
    else:
        y_w = torch.from_numpy(y_wdl).long()
        ds = TensorDataset(X_t, y_v, y_p, y_w)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0, generator=generator)


def _unpack_loader_batch(batch):
    """Normalize dataloader batches to (X, y_value, y_policy, y_wdl_or_none)."""
    if len(batch) == 4:
        X_b, yv_b, yp_b, yw_b = batch
        return X_b, yv_b, yp_b, yw_b
    if len(batch) == 3:
        X_b, yv_b, yp_b = batch
        return X_b, yv_b, yp_b, None
    raise ValueError(f"Unexpected batch tuple length: {len(batch)}")


def _side_labels_from_positions(positions):
    """Return bool array: True for white-to-move, False for black-to-move."""
    turn_plane = positions[:, 0, 0, TURN_LAYER]
    return turn_plane > 0


def _balanced_train_indices(train_idx, positions, seed, black_ratio=0.5):
    """Build side-balanced train indices with optional black-side skew."""
    if len(train_idx) == 0:
        return train_idx
    r = float(black_ratio)
    if not (0.0 < r < 1.0):
        raise ValueError("black_ratio must be in (0, 1)")
    sides = _side_labels_from_positions(positions[train_idx])
    white = train_idx[sides]
    black = train_idx[~sides]
    if len(white) == 0 or len(black) == 0:
        return train_idx

    n = len(train_idx)
    target_black = int(round(n * r))
    target_black = max(1, min(n - 1, target_black))
    target_white = n - target_black
    rng = np.random.default_rng(seed)
    white_bal = rng.choice(white, size=target_white, replace=len(white) < target_white)
    black_bal = rng.choice(black, size=target_black, replace=len(black) < target_black)
    balanced = np.concatenate([white_bal, black_bal]).astype(np.int64, copy=False)
    rng.shuffle(balanced)
    return balanced


def _filter_train_indices_by_result(train_idx, game_results_side, mode):
    """Filter train indices by side-perspective game outcome."""
    m = str(mode)
    if m == "any":
        return train_idx
    if m == "nonloss":
        keep = game_results_side[train_idx] >= 0
        return train_idx[keep]
    if m == "win":
        keep = game_results_side[train_idx] > 0
        return train_idx[keep]
    raise ValueError(f"Unknown result filter mode: {mode}")


def _filter_human_indices_as_black_by_result(train_idx, game_results_white, mode):
    """Filter human-source indices while treating Black as the training side."""
    m = str(mode)
    if m == "any":
        return train_idx
    if m == "nonloss":
        keep = game_results_white[train_idx] <= 0
        return train_idx[keep]
    if m == "win":
        keep = game_results_white[train_idx] < 0
        return train_idx[keep]
    raise ValueError(f"Unknown result filter mode: {mode}")


def _policy_distill_kl(student_logits, teacher_logits, temperature):
    """KL(teacher || student) distillation loss with temperature scaling."""
    t = float(temperature)
    student_logp = F.log_softmax(student_logits / t, dim=1)
    teacher_p = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean") * (t * t)


def _power_loss(pred, target, exponent=VALUE_LOSS_EXPONENT):
    """Power-law loss: mean(|pred - target|^exp). Stockfish uses 2.5."""
    return torch.pow(torch.abs(pred - target), exponent).mean()


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
                 teacher_model=None, distill_value_weight=0.0,
                 distill_policy_weight=0.0, distill_temperature=1.0,
                 use_wdl_head=False, wdl_loss_weight=0.0):
    model.train()
    total_loss = 0.0
    total_val_loss = 0.0
    total_pol_loss = 0.0
    total_wdl_loss = 0.0
    total_distill_value = 0.0
    total_distill_policy = 0.0
    n = 0

    for batch in loader:
        X_b, yv_b, yp_b, yw_b = _unpack_loader_batch(batch)
        X_b = X_b.to(device)
        yv_b = yv_b.to(device)
        yp_b = yp_b.to(device)
        if yw_b is not None:
            yw_b = yw_b.to(device)

        if use_wdl_head:
            value_pred, policy_pred, wdl_logits = model.forward_with_wdl(X_b)
        else:
            value_pred, policy_pred = model(X_b)
            wdl_logits = None
        loss_val = _power_loss(value_pred, yv_b)
        loss_pol = F.cross_entropy(policy_pred, yp_b)
        loss = loss_val + policy_weight * loss_pol
        loss_wdl = torch.zeros((), device=device)
        if (
            use_wdl_head
            and wdl_logits is not None
            and yw_b is not None
            and wdl_loss_weight > 0
        ):
            loss_wdl = F.cross_entropy(wdl_logits, yw_b)
            loss = loss + wdl_loss_weight * loss_wdl

        loss_distill_val = torch.zeros((), device=device)
        loss_distill_pol = torch.zeros((), device=device)
        if teacher_model is not None and (distill_value_weight > 0 or distill_policy_weight > 0):
            with torch.no_grad():
                teacher_value, teacher_policy = teacher_model(X_b)
            if distill_value_weight > 0:
                loss_distill_val = F.mse_loss(value_pred, teacher_value)
                loss = loss + distill_value_weight * loss_distill_val
            if distill_policy_weight > 0:
                loss_distill_pol = _policy_distill_kl(
                    policy_pred, teacher_policy, temperature=distill_temperature,
                )
                loss = loss + distill_policy_weight * loss_distill_pol

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        bs = X_b.size(0)
        total_loss += loss.item() * bs
        total_val_loss += loss_val.item() * bs
        total_pol_loss += loss_pol.item() * bs
        total_wdl_loss += loss_wdl.item() * bs
        total_distill_value += loss_distill_val.item() * bs
        total_distill_policy += loss_distill_pol.item() * bs
        n += bs

    return (
        total_loss / n,
        total_val_loss / n,
        total_pol_loss / n,
        total_wdl_loss / n,
        total_distill_value / n,
        total_distill_policy / n,
    )


@torch.no_grad()
def _eval_epoch(model, loader, device, policy_weight, use_wdl_head=False):
    model.eval()
    total_loss = 0.0
    total_val_loss = 0.0
    total_pol_loss = 0.0
    total_wdl_loss = 0.0
    total_wdl_correct = 0.0
    total_wdl_count = 0
    total_mae = 0.0
    total_mse = 0.0
    n = 0

    for batch in loader:
        X_b, yv_b, yp_b, yw_b = _unpack_loader_batch(batch)
        X_b = X_b.to(device)
        yv_b = yv_b.to(device)
        yp_b = yp_b.to(device)
        if yw_b is not None:
            yw_b = yw_b.to(device)

        if use_wdl_head:
            value_pred, policy_pred, wdl_logits = model.forward_with_wdl(X_b)
        else:
            value_pred, policy_pred = model(X_b)
            wdl_logits = None
        loss_val = _power_loss(value_pred, yv_b)
        loss_pol = F.cross_entropy(policy_pred, yp_b)
        loss = loss_val + policy_weight * loss_pol
        loss_wdl = torch.zeros((), device=device)
        if use_wdl_head and wdl_logits is not None and yw_b is not None:
            loss_wdl = F.cross_entropy(wdl_logits, yw_b)
            preds = torch.argmax(wdl_logits, dim=1)
            total_wdl_correct += (preds == yw_b).sum().item()
            total_wdl_count += int(yw_b.numel())

        bs = X_b.size(0)
        total_loss += loss.item() * bs
        total_val_loss += loss_val.item() * bs
        total_pol_loss += loss_pol.item() * bs
        total_wdl_loss += loss_wdl.item() * bs
        total_mae += (value_pred - yv_b).abs().sum().item()
        total_mse += F.mse_loss(value_pred, yv_b, reduction="sum").item()
        n += bs

    wdl_acc = None
    if total_wdl_count > 0:
        wdl_acc = total_wdl_correct / total_wdl_count
    return (
        total_loss / n,
        total_val_loss / n,
        total_pol_loss / n,
        total_mae / n,
        total_mse / n,
        total_wdl_loss / n,
        wdl_acc,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Monster Chess dual-head network")
    parser.add_argument("--data-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--policy-loss-weight", type=float, default=POLICY_LOSS_WEIGHT,
                        help=f"Weight for policy CE term relative to value loss (default: {POLICY_LOSS_WEIGHT})")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--warmup-start-factor", type=float, default=WARMUP_START_FACTOR)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--use-se-blocks", action=argparse.BooleanOptionalAction, default=USE_SE_BLOCKS,
                        help=f"Enable SE modules in residual blocks (default: {USE_SE_BLOCKS})")
    parser.add_argument("--se-reduction", type=int, default=SE_REDUCTION,
                        help=f"SE channel reduction ratio (default: {SE_REDUCTION})")
    parser.add_argument("--use-side-specialized-heads", action=argparse.BooleanOptionalAction,
                        default=USE_SIDE_SPECIALIZED_HEADS,
                        help=f"Use side-specialized value/policy heads (default: {USE_SIDE_SPECIALIZED_HEADS})")
    parser.add_argument("--balanced-sides-train", action=argparse.BooleanOptionalAction, default=False,
                        help="Use side-balanced white/black-to-move sampling for each train epoch")
    parser.add_argument("--balanced-black-ratio", type=float, default=0.5,
                        help="When balanced sampling is enabled, target fraction of black-to-move samples")
    parser.add_argument("--train-only-side", type=str, default="none",
                        choices=["none", "white", "black"],
                        help="Restrict training samples to one side-to-move")
    parser.add_argument("--train-side-result-filter", type=str, default="any",
                        choices=["any", "nonloss", "win"],
                        help="When training on one side, optionally keep only non-loss or win outcomes for that side")
    parser.add_argument("--black-include-human-source", action=argparse.BooleanOptionalAction, default=False,
                        help="When --train-only-side=black, include human source samples as black-side supervision")
    parser.add_argument("--distill-from", type=str, default=None,
                        help="Teacher checkpoint path for anti-forgetting distillation")
    parser.add_argument("--distill-value-weight", type=float, default=0.0,
                        help="Weight for value distillation MSE term")
    parser.add_argument("--distill-policy-weight", type=float, default=0.0,
                        help="Weight for policy distillation KL term")
    parser.add_argument("--distill-temperature", type=float, default=1.0,
                        help="Temperature for policy distillation")
    parser.add_argument("--target", type=str, default=VALUE_TARGET,
                        choices=["game_result", "mcts_value", "blend", "source_aware"])
    parser.add_argument("--value-head", type=str, default=VALUE_HEAD_MODE,
                        choices=["scalar", "wdl"],
                        help=f"Value head mode (default: {VALUE_HEAD_MODE})")
    parser.add_argument("--wdl-loss-weight", type=float, default=WDL_LOSS_WEIGHT,
                        help=f"WDL CE auxiliary loss weight (default: {WDL_LOSS_WEIGHT})")
    parser.add_argument("--wdl-draw-epsilon", type=float, default=WDL_DRAW_EPSILON,
                        help=f"Draw band for WDL labels, |target|<=eps (default: {WDL_DRAW_EPSILON})")
    args = parser.parse_args()

    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if not (0.0 < args.warmup_start_factor <= 1.0):
        raise ValueError("--warmup-start-factor must be in (0, 1]")
    if args.grad_clip < 0:
        raise ValueError("--grad-clip must be >= 0")
    if args.policy_loss_weight <= 0:
        raise ValueError("--policy-loss-weight must be > 0")
    if args.se_reduction <= 0:
        raise ValueError("--se-reduction must be > 0")
    if args.distill_value_weight < 0:
        raise ValueError("--distill-value-weight must be >= 0")
    if args.distill_policy_weight < 0:
        raise ValueError("--distill-policy-weight must be >= 0")
    if args.distill_temperature <= 0:
        raise ValueError("--distill-temperature must be > 0")
    if (args.distill_value_weight > 0 or args.distill_policy_weight > 0) and not args.distill_from:
        raise ValueError("Distillation weights require --distill-from")
    if not (0.0 < args.balanced_black_ratio < 1.0):
        raise ValueError("--balanced-black-ratio must be in (0, 1)")
    if args.wdl_loss_weight < 0:
        raise ValueError("--wdl-loss-weight must be >= 0")
    if args.wdl_draw_epsilon < 0:
        raise ValueError("--wdl-draw-epsilon must be >= 0")
    if args.value_head == "scalar" and args.wdl_loss_weight > 0:
        print("Warning: --wdl-loss-weight ignored because --value-head=scalar")
    if args.train_only_side == "none" and args.train_side_result_filter != "any":
        print("Warning: --train-side-result-filter ignored because --train-only-side=none")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"Loading data from {args.data_dir}...")
    positions, mcts_values, game_results, policies, target_lambdas, source_ids, splits = load_data(args.data_dir)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Empty split detected (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}). "
            "Regenerate processed data with enough games per split."
        )

    game_results_side = to_side_perspective(game_results, positions)
    wdl_targets = build_wdl_targets(game_results_side, draw_epsilon=args.wdl_draw_epsilon)

    # For non-blend targets, compute once. For blend, recompute per epoch.
    if args.target != "blend":
        value_targets = get_targets(
            mcts_values, game_results_side, args.target, BLEND_WEIGHT,
            target_lambdas=target_lambdas,
        )
    else:
        value_targets = None  # computed per epoch with annealing

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Value target: {args.target}")
    if args.value_head == "wdl":
        losses = int((wdl_targets == 0).sum())
        draws = int((wdl_targets == 1).sum())
        wins = int((wdl_targets == 2).sum())
        print(
            "WDL targets: "
            f"loss={losses} draw={draws} win={wins} "
            f"(draw_epsilon={args.wdl_draw_epsilon:.3f}, ce_w={args.wdl_loss_weight:.3f})"
        )
    if args.target == "source_aware":
        print(
            "Source-aware lambda stats: "
            f"min={target_lambdas.min():.2f} "
            f"max={target_lambdas.max():.2f} "
            f"mean={target_lambdas.mean():.2f}"
        )
    if source_ids is not None:
        human_count = int((source_ids == SOURCE_ID_HUMAN).sum())
        print(f"Source IDs loaded: yes (human samples={human_count})")
    else:
        print("Source IDs loaded: no (legacy processed data format)")
    print(f"Policy loss weight: {args.policy_loss_weight}")
    if args.target == "blend":
        print(f"Lambda annealing: {BLEND_START} -> {BLEND_END} over {args.epochs} epochs")

    # Build model
    model = build_model(
        use_se_blocks=args.use_se_blocks,
        se_reduction=args.se_reduction,
        use_side_specialized_heads=args.use_side_specialized_heads,
        use_wdl_head=(args.value_head == "wdl"),
        value_head_mode=args.value_head,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Policy head channels: {model.policy_head_channels}")
    print(f"Stem channels: {model.stem_channels}")
    print(f"Residual blocks: {list(model.residual_block_channels)}")
    print(f"SE blocks: {model.use_se_blocks} (reduction={model.se_reduction})")
    print(f"Side-specialized heads: {model.use_side_specialized_heads}")
    print(f"Value head mode: {model.value_head_mode} (wdl_head={model.use_wdl_head})")
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

    teacher_model = None
    distill_enabled = args.distill_value_weight > 0 or args.distill_policy_weight > 0
    if args.distill_from:
        if not os.path.exists(args.distill_from):
            raise FileNotFoundError(f"--distill-from not found: {args.distill_from}")
        if distill_enabled:
            teacher_model, _ = load_model_for_inference(args.distill_from, device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            print(
                f"Distillation teacher: {args.distill_from} "
                f"(value_w={args.distill_value_weight}, policy_w={args.distill_policy_weight}, "
                f"T={args.distill_temperature})"
            )
        else:
            print("Distillation teacher path provided, but both distillation weights are zero.")

    optimizer, decay_count, no_decay_count = build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
    )
    print(f"Optimizer: AdamW (weight_decay={args.weight_decay})")
    print(f"  Param groups: decay={decay_count}, no_decay={no_decay_count}")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=LR_GAMMA,
    )

    # Checkpoint setup
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, "best_value_net.pt")
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
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
        "policy_head_channels": int(model.policy_head_channels),
        "stem_channels": int(model.stem_channels),
        "residual_block_channels": [int(x) for x in model.residual_block_channels],
        "residual_block_count": int(model.residual_block_count),
        "use_se_blocks": bool(model.use_se_blocks),
        "se_reduction": int(model.se_reduction),
        "use_side_specialized_heads": bool(model.use_side_specialized_heads),
        "value_head_mode": str(model.value_head_mode),
        "use_wdl_head": bool(model.use_wdl_head),
        "wdl_draw_epsilon": float(args.wdl_draw_epsilon),
        "wdl_loss_weight": float(args.wdl_loss_weight),
        "balanced_sides_train": bool(args.balanced_sides_train),
        "balanced_black_ratio": float(args.balanced_black_ratio),
        "train_only_side": args.train_only_side,
        "distillation": {
            "enabled": bool(distill_enabled),
            "teacher_path": args.distill_from,
            "value_weight": float(args.distill_value_weight),
            "policy_weight": float(args.distill_policy_weight),
            "temperature": float(args.distill_temperature),
        },
        "resume_from": args.resume_from,
        "resume_loaded_tensors": int(resume_loaded_count) if resume_loaded_count is not None else None,
        "resume_skipped_keys": resume_skipped if resume_skipped is not None else [],
        "optimizer": {
            "name": "AdamW",
            "weight_decay": args.weight_decay,
            "decay_group_count": decay_count,
            "no_decay_group_count": no_decay_count,
        },
        "loss_weights": {
            "policy": float(args.policy_loss_weight),
            "value": 1.0,
        },
        "warmup": {
            "epochs": args.warmup_epochs,
            "start_factor": args.warmup_start_factor,
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

    # Training loop
    result_filter_fallback_warned = False
    black_human_include_logged = False
    for epoch in range(1, args.epochs + 1):
        # Recompute blend targets with annealed lambda each epoch
        if args.target == "blend":
            lam = get_blend_lambda(epoch, args.epochs)
            epoch_targets = lam * mcts_values + (1 - lam) * game_results_side
        else:
            epoch_targets = value_targets
            lam = None

        lr_used = _set_epoch_lr(
            optimizer,
            epoch=epoch,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            warmup_start_factor=args.warmup_start_factor,
        )

        train_gen = torch.Generator()
        train_gen.manual_seed(args.seed + epoch)
        base_train_idx = train_idx
        if args.train_only_side != "none":
            all_sides = _side_labels_from_positions(positions[train_idx])
            if args.train_only_side == "white":
                base_train_idx = train_idx[all_sides]
            else:
                base_train_idx = train_idx[~all_sides]
            if len(base_train_idx) == 0:
                raise ValueError(f"--train-only-side={args.train_only_side} produced 0 samples")
            if args.train_side_result_filter != "any":
                filtered_idx = _filter_train_indices_by_result(
                    base_train_idx, game_results_side, args.train_side_result_filter
                )
                if len(filtered_idx) > 0:
                    base_train_idx = filtered_idx
                elif not result_filter_fallback_warned:
                    print(
                        "Warning: --train-side-result-filter="
                        f"{args.train_side_result_filter} yielded 0 samples; using unfiltered side data"
                    )
                    result_filter_fallback_warned = True
            if (
                args.train_only_side == "black"
                and args.black_include_human_source
                and source_ids is not None
            ):
                human_idx = train_idx[source_ids[train_idx] == SOURCE_ID_HUMAN]
                if args.train_side_result_filter != "any":
                    human_idx = _filter_human_indices_as_black_by_result(
                        human_idx, game_results, args.train_side_result_filter
                    )
                if len(human_idx) > 0:
                    base_train_idx = np.unique(
                        np.concatenate([base_train_idx, human_idx]).astype(np.int64, copy=False)
                    )
                    if not black_human_include_logged:
                        print(
                            "Including human source samples while training black: "
                            f"{len(human_idx)} positions"
                        )
                        black_human_include_logged = True
                elif not black_human_include_logged:
                    print(
                        "Warning: --black-include-human-source enabled but no eligible human samples "
                        "matched current filters"
                    )
                    black_human_include_logged = True
        if args.balanced_sides_train:
            epoch_train_idx = _balanced_train_indices(
                base_train_idx, positions, seed=args.seed + epoch * 17,
                black_ratio=args.balanced_black_ratio,
            )
        else:
            epoch_train_idx = base_train_idx
        val_loader = _make_loader(
            positions[val_idx],
            epoch_targets[val_idx],
            policies[val_idx],
            args.batch_size,
            shuffle=False,
            y_wdl=wdl_targets[val_idx] if args.value_head == "wdl" else None,
        )

        # Keep policy labels aligned with selected train indices.
        train_loader = _make_loader(
            positions[epoch_train_idx],
            epoch_targets[epoch_train_idx],
            policies[epoch_train_idx],
            args.batch_size,
            shuffle=True,
            generator=train_gen,
            y_wdl=wdl_targets[epoch_train_idx] if args.value_head == "wdl" else None,
        )

        train_loss, train_v, train_p, train_wdl, train_dv, train_dp = _train_epoch(
            model, train_loader, optimizer, device, args.policy_loss_weight, args.grad_clip,
            teacher_model=teacher_model,
            distill_value_weight=args.distill_value_weight,
            distill_policy_weight=args.distill_policy_weight,
            distill_temperature=args.distill_temperature,
            use_wdl_head=(args.value_head == "wdl"),
            wdl_loss_weight=args.wdl_loss_weight if args.value_head == "wdl" else 0.0,
        )
        val_loss, val_v, val_p, val_mae, val_mse, val_wdl, val_wdl_acc = _eval_epoch(
            model, val_loader, device, args.policy_loss_weight,
            use_wdl_head=(args.value_head == "wdl"),
        )
        if epoch > args.warmup_epochs:
            scheduler.step()

        lr = lr_used
        lam_str = f"  λ={lam:.2f}" if (args.target == "blend" and lam is not None) else ""
        distill_str = ""
        if distill_enabled:
            distill_str = f"  distill(v={train_dv:.4f} p={train_dp:.4f})"
        wdl_str = ""
        if args.value_head == "wdl":
            acc_str = f"{val_wdl_acc:.1%}" if val_wdl_acc is not None else "n/a"
            wdl_str = f"  wdl(train_ce={train_wdl:.4f} val_ce={val_wdl:.4f} val_acc={acc_str})"
        balance_str = ""
        if args.balanced_sides_train:
            balance_str = f" balanced(b={args.balanced_black_ratio:.2f})"
        side_str = ""
        if args.train_only_side != "none":
            side_str = f" side={args.train_only_side}"
        print(f"Epoch {epoch:3d}  "
              f"train={train_loss:.4f} (v={train_v:.4f} p={train_p:.4f})  "
              f"val={val_loss:.4f} (pow={val_v:.4f} mse={val_mse:.4f} p={val_p:.4f} mae={val_mae:.4f})  "
              f"lr={lr:.1e}{lam_str}{distill_str}{wdl_str}{balance_str}{side_str}")
        run_metadata["epochs"].append({
            "epoch": epoch,
            "train_samples": int(len(epoch_train_idx)),
            "train_total_loss": float(train_loss),
            "train_value_power_loss": float(train_v),
            "train_policy_ce": float(train_p),
            "train_wdl_ce": float(train_wdl),
            "train_distill_value_mse": float(train_dv),
            "train_distill_policy_kl": float(train_dp),
            "val_total_loss": float(val_loss),
            "val_value_power_loss": float(val_v),
            "val_value_mse": float(val_mse),
            "val_policy_ce": float(val_p),
            "val_value_mae": float(val_mae),
            "val_wdl_ce": float(val_wdl),
            "val_wdl_accuracy": float(val_wdl_acc) if val_wdl_acc is not None else None,
            "lr": float(lr),
            "blend_lambda": float(lam) if lam is not None else None,
        })

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model for test evaluation
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    # Use final epoch targets for test eval
    final_targets = epoch_targets if args.target == "blend" else value_targets
    test_loader = _make_loader(
        positions[test_idx],
        final_targets[test_idx],
        policies[test_idx],
        args.batch_size,
        shuffle=False,
        y_wdl=wdl_targets[test_idx] if args.value_head == "wdl" else None,
    )
    test_loss, test_v, test_p, test_mae, test_mse, test_wdl, test_wdl_acc = _eval_epoch(
        model, test_loader, device, args.policy_loss_weight,
        use_wdl_head=(args.value_head == "wdl"),
    )

    print("\n--- Test set evaluation ---")
    print(f"Total loss: {test_loss:.4f}")
    print(f"Value power loss: {test_v:.4f}")
    print(f"Value true MSE:   {test_mse:.4f}")
    print(f"Policy CE:  {test_p:.4f}")
    print(f"Value MAE:  {test_mae:.4f}")
    if args.value_head == "wdl":
        acc_str = f"{test_wdl_acc:.1%}" if test_wdl_acc is not None else "n/a"
        print(f"WDL CE:     {test_wdl:.4f}")
        print(f"WDL Acc:    {acc_str}")

    # Value sign-accuracy
    model.eval()
    all_vpreds = []
    all_vtrue = []
    with torch.no_grad():
        for batch in test_loader:
            X_b, yv_b, _, _ = _unpack_loader_batch(batch)
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
        "winner_sign_accuracy_non_draw": float(acc) if non_draw.sum() > 0 else None,
    }
    run_metadata["best_val_loss"] = float(best_val_loss)
    run_metadata["checkpoint_path"] = checkpoint_path
    run_metadata["end_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"\nBest model saved to {checkpoint_path}")
    print(f"Run metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
