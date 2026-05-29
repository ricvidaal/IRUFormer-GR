import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, Concatenate, Add, Multiply,
    GlobalAveragePooling2D, Dense, Reshape, LayerNormalization,
    MultiHeadAttention, Activation
)
from tensorflow.keras.models import Model

def squeeze_excite_block(x, ratio=8, name=None):
    channels = x.shape[-1]
    if channels is None:
        raise ValueError("Channel dimension must be defined for SE block.")

    se = GlobalAveragePooling2D(name=None if name is None else f"{name}_gap")(x)
    se = Dense(max(channels // ratio, 4), activation="relu",
               name=None if name is None else f"{name}_fc1")(se)
    se = Dense(channels, activation="sigmoid",
               name=None if name is None else f"{name}_fc2")(se)
    se = Reshape((1, 1, channels), name=None if name is None else f"{name}_reshape")(se)
    return Multiply(name=None if name is None else f"{name}_scale")([x, se])


def inception_block(x, filters, use_se=True, name=None):
    branch_a = Conv2D(
        filters, (3, 3), padding="same", activation="relu",
        kernel_initializer="he_uniform",
        name=None if name is None else f"{name}_b1_conv"
    )(x)

    branch_b = Conv2D(
        filters * 2, (3, 3), padding="same", activation="relu",
        kernel_initializer="he_uniform",
        name=None if name is None else f"{name}_b2_conv"
    )(x)

    branch_c = Conv2D(
        filters, (3, 3), padding="same", activation="relu",
        kernel_initializer="he_uniform", dilation_rate=2,
        name=None if name is None else f"{name}_b3_conv"
    )(x)

    concat = Concatenate(name=None if name is None else f"{name}_concat")([branch_a, branch_b, branch_c])
    reduced = Conv2D(
        filters, (1, 1), padding="same",
        name=None if name is None else f"{name}_reduce"
    )(concat)

    if use_se:
        reduced = squeeze_excite_block(reduced, ratio=8, name=None if name is None else f"{name}_se")

    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(
            filters, (1, 1), padding="same",
            name=None if name is None else f"{name}_shortcut"
        )(x)

    out = Add(name=None if name is None else f"{name}_add")([shortcut, reduced])
    return out


def inception_block_reduction(x, filters, use_se=True, name=None):
    shortcut = Conv2D(
        filters, (2, 2), strides=2, padding="same",
        name=None if name is None else f"{name}_shortcut"
    )(x)

    branch_a = Conv2D(
        filters, (3, 3), strides=2, padding="same", activation="relu",
        kernel_initializer="he_uniform",
        name=None if name is None else f"{name}_b1_conv"
    )(x)

    branch_b = Conv2D(
        filters * 2, (3, 3), strides=2, padding="same", activation="relu",
        kernel_initializer="he_uniform",
        name=None if name is None else f"{name}_b2_conv"
    )(x)

    branch_c = tf.keras.layers.AveragePooling2D(
        (2, 2), strides=2, padding="same",
        name=None if name is None else f"{name}_b3_pool"
    )(x)
    branch_c = Conv2D(
        filters, (1, 1), padding="same",
        name=None if name is None else f"{name}_b3_proj"
    )(branch_c)

    concat = Concatenate(name=None if name is None else f"{name}_concat")([branch_a, branch_b, branch_c])
    reduced = Conv2D(
        filters, (1, 1), padding="same",
        name=None if name is None else f"{name}_reduce"
    )(concat)

    if use_se:
        reduced = squeeze_excite_block(reduced, ratio=8, name=None if name is None else f"{name}_se")

    out = Add(name=None if name is None else f"{name}_add")([shortcut, reduced])
    return out


def skip_fusion(skip, upsampled, filters, name=None):
    x = Concatenate(name=None if name is None else f"{name}_concat")([skip, upsampled])
    x = Conv2D(
        filters, (1, 1), padding="same", activation="relu",
        kernel_initializer="he_uniform",
        name=None if name is None else f"{name}_fuse"
    )(x)
    return x


def transformer_block_2d(x, num_heads=4, key_dim=16, ff_mult=4, name=None):
    channels = x.shape[-1]
    if channels is None:
        raise ValueError("Channel dimension must be defined for transformer block.")

    # Flatten spatial dims -> tokens
    h = tf.keras.layers.Reshape((-1, channels), name=None if name is None else f"{name}_reshape_in")(x)

    # MHSA + residual
    h1 = LayerNormalization(epsilon=1e-6, name=None if name is None else f"{name}_ln1")(h)
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name=None if name is None else f"{name}_mha"
    )(h1, h1)
    h = Add(name=None if name is None else f"{name}_attn_add")([h, attn])

    # FFN + residual
    h2 = LayerNormalization(epsilon=1e-6, name=None if name is None else f"{name}_ln2")(h)
    ff = Dense(channels * ff_mult, activation="gelu",
               name=None if name is None else f"{name}_ff1")(h2)
    ff = Dense(channels, name=None if name is None else f"{name}_ff2")(ff)
    h = Add(name=None if name is None else f"{name}_ff_add")([h, ff])

    # Restore spatial shape
    out = tf.keras.layers.Reshape(
        (x.shape[1], x.shape[2], channels),
        name=None if name is None else f"{name}_reshape_out"
    )(h)
    return out


def IRUFormer_GR(
    H=96,
    W=96,
    C=3,
    base_filters=16,
    transformer_blocks=2,
    num_heads=4,
    key_dim=16,
    use_se=True
):
    inputs = Input(shape=(H, W, C), name="input_image")

    # Head
    head = Conv2D(
        base_filters, (3, 3), padding="same", activation="relu",
        kernel_initializer="he_uniform", name="head_conv"
    )(inputs)

    # Encoder
    conv1 = inception_block_reduction(head, base_filters, use_se=use_se, name="enc1_down")
    conv1 = inception_block(conv1, base_filters, use_se=use_se, name="enc1_block")

    conv2 = inception_block_reduction(conv1, base_filters * 2, use_se=use_se, name="enc2_down")
    conv2 = inception_block(conv2, base_filters * 2, use_se=use_se, name="enc2_block")

    conv3 = inception_block_reduction(conv2, base_filters * 4, use_se=use_se, name="enc3_down")
    conv3 = inception_block(conv3, base_filters * 4, use_se=use_se, name="enc3_block")

    # Bottleneck
    body = inception_block_reduction(conv3, base_filters * 8, use_se=use_se, name="bottleneck_down")
    body = inception_block(body, base_filters * 8, use_se=use_se, name="bottleneck_block")

    for i in range(transformer_blocks):
        body = transformer_block_2d(
            body,
            num_heads=num_heads,
            key_dim=key_dim,
            ff_mult=4,
            name=f"bottleneck_transformer_{i+1}"
        )

    # Decoder
    deconv3 = Conv2DTranspose(
        base_filters * 4, (2, 2), strides=2, padding="same",
        activation="relu", kernel_initializer="he_uniform",
        name="dec3_up"
    )(body)
    deconv3 = skip_fusion(conv3, deconv3, base_filters * 4, name="dec3_skip")
    deconv3 = inception_block(deconv3, base_filters * 4, use_se=use_se, name="dec3_block")

    deconv2 = Conv2DTranspose(
        base_filters * 2, (2, 2), strides=2, padding="same",
        activation="relu", kernel_initializer="he_uniform",
        name="dec2_up"
    )(deconv3)
    deconv2 = skip_fusion(conv2, deconv2, base_filters * 2, name="dec2_skip")
    deconv2 = inception_block(deconv2, base_filters * 2, use_se=use_se, name="dec2_block")

    deconv1 = Conv2DTranspose(
        base_filters, (2, 2), strides=2, padding="same",
        activation="relu", kernel_initializer="he_uniform",
        name="dec1_up"
    )(deconv2)
    deconv1 = skip_fusion(conv1, deconv1, base_filters, name="dec1_skip")
    deconv1 = inception_block(deconv1, base_filters, use_se=use_se, name="dec1_block")

    tail = Conv2DTranspose(
        base_filters, (2, 2), strides=2, padding="same",
        activation="relu", kernel_initializer="he_uniform",
        name="tail_up"
    )(deconv1)
    tail = inception_block(tail, base_filters, use_se=use_se, name="tail_block")

    # Residual global output
    residual = Conv2D(
        C, (1, 1), padding="same",
        activation=None, name="residual_pred"
    )(tail)

    output = Add(name="global_residual_add")([inputs, residual])
    output = Activation("sigmoid", name="output_image")(output)

    return Model(inputs, output, name="IRUFormer_GR")
