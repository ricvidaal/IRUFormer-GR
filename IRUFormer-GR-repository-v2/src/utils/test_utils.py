import os
import numpy as np
import tifffile as tiff
from pathlib import Path
import csv

def add_gaussian_noise_to_test_set(
    input_dir,
    output_dir,
    min_sigma=1,
    max_sigma=50,
    seed=0,
    shuffle_noise_levels=True,
    save_manifest=True
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    image_paths = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )

    if len(image_paths) == 0:
        raise ValueError(f"No se encontraron archivos .tif/.tiff en {input_dir}")

    sigmas = np.linspace(min_sigma, max_sigma, len(image_paths))

    if shuffle_noise_levels:
        rng.shuffle(sigmas)

    manifest_rows = []

    for image_path, sigma in zip(image_paths, sigmas):
        image = tiff.imread(image_path)
        original_dtype = image.dtype
        image_float = image.astype(np.float32)

        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=image.shape
        ).astype(np.float32)

        noisy_image = image_float + noise

        if np.issubdtype(original_dtype, np.integer):
            dtype_info = np.iinfo(original_dtype)
            noisy_image = np.clip(
                noisy_image,
                dtype_info.min,
                dtype_info.max
            )
            noisy_image = noisy_image.astype(original_dtype)
        else:
            noisy_image = noisy_image.astype(original_dtype)

        output_path = output_dir / image_path.name
        tiff.imwrite(output_path, noisy_image)

        manifest_rows.append({
            "filename": image_path.name,
            "sigma": float(sigma),
            "output_path": str(output_path)
        })

    if save_manifest:
        manifest_path = output_dir / "noise_manifest.csv"
        with open(manifest_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename", "sigma", "output_path"]
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"Procesadas {len(image_paths)} imágenes.")

def get_tif_paths(folder):
    folder = Path(folder)
    image_paths = sorted(
        list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    )
    return image_paths

def select_random_test_images(
    image_paths,
    num_images=57458, # Updated to 57458 as requested
    seed=0
):
    if len(image_paths) < num_images:
        print(f"Warning: Only {len(image_paths)} images available. Selecting all.")
        num_images = len(image_paths)

    rng = np.random.default_rng(seed)
    selected_paths = rng.choice(
        image_paths,
        size=num_images,
        replace=False
    )
    return list(selected_paths)

def load_image_normalized(path):
    img = tiff.imread(path)
    img = img.astype(np.float32)
    img = img / 255.0
    return img

def save_uint8_tif(path, image):
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    tiff.imwrite(path, image)
