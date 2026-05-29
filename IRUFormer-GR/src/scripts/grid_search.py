import os
import tensorflow as tf
import itertools
import pandas as pd
import gc
import sys
from pathlib import Path

# Add the project root to sys.path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Corrected Imports from src package
from src.models.iruformer_gr import IRUFormer_GR
from src.utils.data_loaders import make_train_val_datasets_from_folder_lazy
from src.utils.metrics import (
    charbonnier_loss, ssim_loss, combined_denoising_loss,
    psnr_metric, ssim_metric
)

def build_and_compile_model(H, W, C,
    base_filters,
    transformer_blocks,
    num_heads,
    key_dim,
    use_se,
    learning_rate
):
    model = IRUFormer_GR(
        H=H,
        W=W,
        C=C,
        base_filters=base_filters,
        transformer_blocks=transformer_blocks,
        num_heads=num_heads,
        key_dim=key_dim,
        use_se=use_se
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=combined_denoising_loss,
        metrics=[
            psnr_metric,
            ssim_metric
        ]
    )

    return model

def create_grid_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = []
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        combinations.append(params)
    return combinations

def run_grid_search(
    train_ds,
    val_ds,
    param_grid,
    H=96,
    W=96,
    C=3,
    epochs=30,
    patience=5,
    batch_size=16,
    use_se=True,
    results_csv_path="../../results/grid_search_results.csv",
    models_dir="../../results/models"
):
    os.makedirs(models_dir, exist_ok=True)
    combinations = create_grid_combinations(param_grid)
    results = []

    best_val_psnr = -float("inf")
    best_params = None
    best_model_path = None

    print(f"Total combinations to test: {len(combinations)}")

    for index, params in enumerate(combinations):
        print("\n" + "=" * 80)
        print(f"Testing combination {index + 1}/{len(combinations)}")
        print(params)
        print("=" * 80)

        tf.keras.backend.clear_session()
        gc.collect()

        model = build_and_compile_model(
            H=H, W=W, C=C,
            base_filters=params["base_filters"],
            transformer_blocks=params["transformer_blocks"],
            num_heads=params["num_heads"],
            key_dim=params["key_dim"],
            use_se=use_se,
            learning_rate=params["learning_rate"]
        )

        model_path = os.path.join(models_dir, f"model_{index + 1}.keras")

        callbacks = [
            EarlyStopping(monitor="val_psnr_metric", mode="max", patience=patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_psnr_metric", mode="max", factor=0.5, patience=max(2, patience // 2), min_lr=1e-7, verbose=1),
            ModelCheckpoint(filepath=model_path, monitor="val_psnr_metric", mode="max", save_best_only=True, verbose=0)
        ]

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

        max_val_psnr = max(history.history["val_psnr_metric"])
        best_epoch_index = history.history["val_psnr_metric"].index(max_val_psnr)

        result = {
            "combination": index + 1,
            "base_filters": params["base_filters"],
            "transformer_blocks": params["transformer_blocks"],
            "num_heads": params["num_heads"],
            "key_dim": params["key_dim"],
            "learning_rate": params["learning_rate"],
            "best_epoch": best_epoch_index + 1,
            "train_psnr_metric": history.history["psnr_metric"][best_epoch_index],
            "val_psnr_metric": max_val_psnr,
            "train_ssim_metric": history.history["ssim_metric"][best_epoch_index],
            "val_ssim_metric": history.history["val_ssim_metric"][best_epoch_index],
            "model_path": model_path
        }
        results.append(result)
        pd.DataFrame(results).to_csv(results_csv_path, index=False)

        if max_val_psnr > best_val_psnr:
            best_val_psnr = max_val_psnr
            best_params = params
            best_model_path = model_path
            print(f"New best model found! PSNR: {best_val_psnr}")

    return pd.DataFrame(results), best_params, best_model_path

if __name__ == "__main__":
    # Example usage
    train_ds, val_ds = make_train_val_datasets_from_folder_lazy(
        dataset_dir="../../data/train", # Adjust path as needed
        img_size=(96, 96),
        batch_size=16
    )

    param_grid = {
        "base_filters": [8, 16],
        "transformer_blocks": [1, 2, 3],
        "num_heads": [2, 4],
        "key_dim": [16, 24],
        "learning_rate": [1e-4]
    }

    run_grid_search(train_ds, val_ds, param_grid)
