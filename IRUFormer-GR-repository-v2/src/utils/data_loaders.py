import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize

def get_pair_paths_alternating_from_folder(dataset_dir):
    dataset_dir = Path(dataset_dir)

    image_paths = sorted([
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ])

    if len(image_paths) < 2:
        raise ValueError(f"No hay suficientes imágenes en {dataset_dir}")

    if len(image_paths) % 2 != 0:
        print(f"⚠️ Número impar de imágenes en {dataset_dir}. Se ignorará la última.")
        image_paths = image_paths[:-1]

    noisy_paths = []
    clean_paths = []

    for i in range(0, len(image_paths), 2):
        clean_path = image_paths[i]
        noise_path = image_paths[i + 1]

        clean_paths.append(str(clean_path))
        noisy_paths.append(str(noise_path))

    print(f"Pares encontrados: {len(noisy_paths)}")

    return noisy_paths, clean_paths


def normalize_image(img):
    img = img.astype(np.float32)

    max_value = np.max(img)

    if max_value > 255:
        img = img / 65535.0
    elif max_value > 1:
        img = img / 255.0

    img = np.clip(img, 0.0, 1.0)

    return img


def load_tif_pair_with_skimage(noisy_path, clean_path, img_size=(96, 96)):
    noisy_path = noisy_path.decode("utf-8")
    clean_path = clean_path.decode("utf-8")

    noisy = imread(noisy_path)
    clean = imread(clean_path)

    noisy = resize(
        noisy,
        img_size,
        preserve_range=True,
        anti_aliasing=True
    )

    clean = resize(
        clean,
        img_size,
        preserve_range=True,
        anti_aliasing=True
    )

    noisy = normalize_image(noisy)
    clean = normalize_image(clean)

    if noisy.ndim == 2:
        noisy = np.stack([noisy] * 3, axis=-1)

    if clean.ndim == 2:
        clean = np.stack([clean] * 3, axis=-1)

    if noisy.shape[-1] > 3:
        noisy = noisy[..., :3]

    if clean.shape[-1] > 3:
        clean = clean[..., :3]

    noisy = noisy.astype(np.float32)
    clean = clean.astype(np.float32)

    return noisy, clean


def tf_load_tif_pair(noisy_path, clean_path, img_size=(96, 96)):
    noisy, clean = tf.numpy_function(
        func=lambda n, c: load_tif_pair_with_skimage(n, c, img_size),
        inp=[noisy_path, clean_path],
        Tout=[tf.float32, tf.float32]
    )

    noisy.set_shape((img_size[0], img_size[1], 3))
    clean.set_shape((img_size[0], img_size[1], 3))

    return noisy, clean


def make_train_val_datasets_from_folder_lazy(
    dataset_dir,
    img_size=(96, 96),
    batch_size=16,
    val_split=0.1,
    shuffle=True,
    seed=42,
    num_parallel_calls=2
):
    noisy_paths, clean_paths = get_pair_paths_alternating_from_folder(dataset_dir)

    noisy_paths = np.array(noisy_paths)
    clean_paths = np.array(clean_paths)

    num_samples = len(noisy_paths)

    if num_samples < 2:
        raise ValueError("Necesitas al menos 2 pares de imágenes para crear train y validation.")

    indices = np.arange(num_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    val_size = int(num_samples * val_split)

    if val_size < 1:
        val_size = 1

    train_size = num_samples - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_noisy_paths = noisy_paths[train_indices]
    train_clean_paths = clean_paths[train_indices]

    val_noisy_paths = noisy_paths[val_indices]
    val_clean_paths = clean_paths[val_indices]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_noisy_paths, train_clean_paths)
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_noisy_paths, val_clean_paths)
    )

    if shuffle:
        train_ds = train_ds.shuffle(
            buffer_size=min(train_size, 512),
            seed=seed,
            reshuffle_each_iteration=True
        )

    train_ds = train_ds.map(
        lambda noisy_path, clean_path: tf_load_tif_pair(
            noisy_path,
            clean_path,
            img_size=img_size
        ),
        num_parallel_calls=num_parallel_calls
    )

    val_ds = val_ds.map(
        lambda noisy_path, clean_path: tf_load_tif_pair(
            noisy_path,
            clean_path,
            img_size=img_size
        ),
        num_parallel_calls=num_parallel_calls
    )

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.prefetch(1)
    val_ds = val_ds.prefetch(1)

    print(f"Total pares: {num_samples}")
    print(f"Train pares: {train_size}")
    print(f"Validation pares: {val_size}")
    print(f"Batch size: {batch_size}")

    return train_ds, val_ds
