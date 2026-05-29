import tensorflow as tf

def charbonnier_loss(y_true, y_pred, epsilon=1e-3):
    diff = y_true - y_pred
    return tf.reduce_mean(tf.sqrt(tf.square(diff) + epsilon**2))

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_denoising_loss(y_true, y_pred):
    return 0.8 * charbonnier_loss(y_true, y_pred) + 0.2 * ssim_loss(y_true, y_pred)

def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
