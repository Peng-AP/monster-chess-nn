import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    TENSOR_SHAPE, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    VALUE_TARGET, BLEND_WEIGHT, PROCESSED_DATA_DIR, MODEL_DIR,
)


def build_model():
    """CNN value network: (8, 8, 15) -> scalar in [-1, 1]."""
    inputs = keras.Input(shape=TENSOR_SHAPE)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="tanh")(x)

    model = keras.Model(inputs=inputs, outputs=output)
    return model


def load_data(data_dir):
    """Load processed training data and splits."""
    positions = np.load(os.path.join(data_dir, "positions.npy"))
    mcts_values = np.load(os.path.join(data_dir, "mcts_values.npy"))
    game_results = np.load(os.path.join(data_dir, "game_results.npy"))
    splits = np.load(os.path.join(data_dir, "splits.npz"))

    return positions, mcts_values, game_results, splits


def get_targets(mcts_values, game_results, target_type, blend_weight):
    """Build training targets based on the chosen strategy."""
    if target_type == "game_result":
        return game_results
    elif target_type == "mcts_value":
        return mcts_values
    elif target_type == "blend":
        return blend_weight * mcts_values + (1 - blend_weight) * game_results
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def main():
    parser = argparse.ArgumentParser(description="Train Monster Chess value network")
    parser.add_argument("--data-dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--target", type=str, default=VALUE_TARGET,
                        choices=["game_result", "mcts_value", "blend"])
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_dir}...")
    positions, mcts_values, game_results, splits = load_data(args.data_dir)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    targets = get_targets(mcts_values, game_results, args.target, BLEND_WEIGHT)

    X_train, y_train = positions[train_idx], targets[train_idx]
    X_val, y_val = positions[val_idx], targets[val_idx]
    X_test, y_test = positions[test_idx], targets[test_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Target: {args.target}")

    # Build and compile
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    # Callbacks
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, "best_value_net.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6,
        ),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    print("\n--- Test set evaluation ---")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Classification accuracy (sign match = correct winner prediction)
    preds = model.predict(X_test, verbose=0).flatten()
    correct = np.sum(np.sign(preds) == np.sign(y_test))
    total = len(y_test)
    # Exclude draws (result=0) from accuracy calc
    non_draw = y_test != 0
    if non_draw.sum() > 0:
        acc = np.sum(np.sign(preds[non_draw]) == np.sign(y_test[non_draw])) / non_draw.sum()
        print(f"Winner prediction accuracy (non-draw): {acc:.1%}")

    print(f"\nBest model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
