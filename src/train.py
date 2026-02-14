import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from config import (
    TENSOR_SHAPE, POLICY_SIZE, POLICY_LOSS_WEIGHT,
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    VALUE_TARGET, BLEND_WEIGHT, BLEND_START, BLEND_END,
    PROCESSED_DATA_DIR, MODEL_DIR,
    VALUE_LOSS_EXPONENT, LR_GAMMA,
)

# Input channels = last dim of TENSOR_SHAPE (8, 8, 15)
IN_CHANNELS = TENSOR_SHAPE[2]


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
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
        return F.relu(x + shortcut)


class DualHeadNet(nn.Module):
    """Dual-head ResNet: (N, 15, 8, 8) -> (value in [-1,1], policy logits[4096]).

    Shared backbone:
      Conv 64 stem + 2x res blocks (64) + 2x res blocks (128)
    Value head:
      GAP -> Dense 128 -> Dense 64 -> Dense 1 (tanh)
    Policy head:
      Conv 2x1x1 -> BN -> ReLU -> Flatten -> Dense 4096 (logits)
    """
    def __init__(self):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Residual tower
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.res3 = ResidualBlock(64, 128)
        self.res4 = ResidualBlock(128, 128)

        # Value head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (N, 128, 1, 1)
            nn.Flatten(),                  # (N, 128)
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(128, 2, 1, bias=False),  # channel reduction
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * 8 * 8, POLICY_SIZE)

    def forward(self, x):
        # x: (N, C, 8, 8)  — PyTorch uses channels-first
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        value = self.value_head(x)          # (N, 1)
        p = self.policy_conv(x)             # (N, 2, 8, 8)
        p = p.flatten(1)                    # (N, 128)
        policy = self.policy_fc(p)          # (N, 4096)

        return value, policy


def build_model():
    return DualHeadNet()


def load_data(data_dir):
    """Load processed training data and splits."""
    positions = np.load(os.path.join(data_dir, "positions.npy"))
    mcts_values = np.load(os.path.join(data_dir, "mcts_values.npy"))
    game_results = np.load(os.path.join(data_dir, "game_results.npy"))
    policies = np.load(os.path.join(data_dir, "policies.npy"))
    splits = np.load(os.path.join(data_dir, "splits.npz"))
    return positions, mcts_values, game_results, policies, splits


def get_targets(mcts_values, game_results, target_type, blend_weight):
    """Build value training targets based on the chosen strategy."""
    if target_type == "game_result":
        return game_results
    elif target_type == "mcts_value":
        return mcts_values
    elif target_type == "blend":
        return blend_weight * mcts_values + (1 - blend_weight) * game_results
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def get_blend_lambda(epoch, max_epochs):
    """Anneal blend weight from BLEND_START to BLEND_END over training."""
    if max_epochs <= 1:
        return BLEND_START
    t = (epoch - 1) / (max_epochs - 1)
    return BLEND_START + (BLEND_END - BLEND_START) * t


def _make_loader(X, y_val, y_pol, batch_size, shuffle=True):
    """Create a DataLoader from numpy arrays.

    Transposes X from (N, 8, 8, C) to (N, C, 8, 8) for PyTorch.
    """
    X_t = torch.from_numpy(X.transpose(0, 3, 1, 2))  # channels-first
    y_v = torch.from_numpy(y_val).unsqueeze(1)         # (N, 1)
    y_p = torch.from_numpy(y_pol)                      # (N, 4096)
    ds = TensorDataset(X_t, y_v, y_p)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0)


def _power_loss(pred, target, exponent=VALUE_LOSS_EXPONENT):
    """Power-law loss: mean(|pred - target|^exp). Stockfish uses 2.5."""
    return torch.pow(torch.abs(pred - target), exponent).mean()


def _train_epoch(model, loader, optimizer, device, policy_weight):
    model.train()
    total_loss = 0.0
    total_val_loss = 0.0
    total_pol_loss = 0.0
    n = 0

    for X_b, yv_b, yp_b in loader:
        X_b = X_b.to(device)
        yv_b = yv_b.to(device)
        yp_b = yp_b.to(device)

        value_pred, policy_pred = model(X_b)
        loss_val = _power_loss(value_pred, yv_b)
        loss_pol = F.cross_entropy(policy_pred, yp_b)
        loss = loss_val + policy_weight * loss_pol

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = X_b.size(0)
        total_loss += loss.item() * bs
        total_val_loss += loss_val.item() * bs
        total_pol_loss += loss_pol.item() * bs
        n += bs

    return total_loss / n, total_val_loss / n, total_pol_loss / n


@torch.no_grad()
def _eval_epoch(model, loader, device, policy_weight):
    model.eval()
    total_loss = 0.0
    total_val_loss = 0.0
    total_pol_loss = 0.0
    total_mae = 0.0
    n = 0

    for X_b, yv_b, yp_b in loader:
        X_b = X_b.to(device)
        yv_b = yv_b.to(device)
        yp_b = yp_b.to(device)

        value_pred, policy_pred = model(X_b)
        loss_val = _power_loss(value_pred, yv_b)
        loss_pol = F.cross_entropy(policy_pred, yp_b)
        loss = loss_val + policy_weight * loss_pol

        bs = X_b.size(0)
        total_loss += loss.item() * bs
        total_val_loss += loss_val.item() * bs
        total_pol_loss += loss_pol.item() * bs
        total_mae += (value_pred - yv_b).abs().sum().item()
        n += bs

    return total_loss / n, total_val_loss / n, total_pol_loss / n, total_mae / n


def main():
    parser = argparse.ArgumentParser(description="Train Monster Chess dual-head network")
    parser.add_argument("--data-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--target", type=str, default=VALUE_TARGET,
                        choices=["game_result", "mcts_value", "blend"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"Loading data from {args.data_dir}...")
    positions, mcts_values, game_results, policies, splits = load_data(args.data_dir)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # For non-blend targets, compute once. For blend, recompute per epoch.
    if args.target != "blend":
        value_targets = get_targets(mcts_values, game_results, args.target, BLEND_WEIGHT)
    else:
        value_targets = None  # computed per epoch with annealing

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Value target: {args.target}")
    if args.target == "blend":
        print(f"Lambda annealing: {BLEND_START} -> {BLEND_END} over {args.epochs} epochs")

    # Build model
    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=LR_GAMMA,
    )

    # Checkpoint setup
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, "best_value_net.pt")
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Recompute blend targets with annealed lambda each epoch
        if args.target == "blend":
            lam = get_blend_lambda(epoch, args.epochs)
            epoch_targets = lam * mcts_values + (1 - lam) * game_results
        else:
            epoch_targets = value_targets
            lam = BLEND_WEIGHT

        train_loader = _make_loader(positions[train_idx], epoch_targets[train_idx],
                                    policies[train_idx], args.batch_size, shuffle=True)
        val_loader = _make_loader(positions[val_idx], epoch_targets[val_idx],
                                  policies[val_idx], args.batch_size, shuffle=False)

        train_loss, train_v, train_p = _train_epoch(
            model, train_loader, optimizer, device, POLICY_LOSS_WEIGHT,
        )
        val_loss, val_v, val_p, val_mae = _eval_epoch(
            model, val_loader, device, POLICY_LOSS_WEIGHT,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        lam_str = f"  λ={lam:.2f}" if args.target == "blend" else ""
        print(f"Epoch {epoch:3d}  "
              f"train={train_loss:.4f} (v={train_v:.4f} p={train_p:.4f})  "
              f"val={val_loss:.4f} (v={val_v:.4f} p={val_p:.4f} mae={val_mae:.4f})  "
              f"lr={lr:.1e}{lam_str}")

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
    test_loader = _make_loader(positions[test_idx], final_targets[test_idx],
                               policies[test_idx], args.batch_size, shuffle=False)
    test_loss, test_v, test_p, test_mae = _eval_epoch(
        model, test_loader, device, POLICY_LOSS_WEIGHT,
    )

    print("\n--- Test set evaluation ---")
    print(f"Total loss: {test_loss:.4f}")
    print(f"Value MSE:  {test_v:.4f}")
    print(f"Policy CE:  {test_p:.4f}")
    print(f"Value MAE:  {test_mae:.4f}")

    # Value sign-accuracy
    model.eval()
    all_vpreds = []
    all_vtrue = []
    with torch.no_grad():
        for X_b, yv_b, _ in test_loader:
            vp, _ = model(X_b.to(device))
            all_vpreds.append(vp.cpu().numpy())
            all_vtrue.append(yv_b.numpy())
    vpreds = np.concatenate(all_vpreds).flatten()
    vtrue = np.concatenate(all_vtrue).flatten()
    non_draw = vtrue != 0
    if non_draw.sum() > 0:
        acc = np.mean(np.sign(vpreds[non_draw]) == np.sign(vtrue[non_draw]))
        print(f"Winner prediction accuracy (non-draw): {acc:.1%}")

    print(f"\nBest model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
