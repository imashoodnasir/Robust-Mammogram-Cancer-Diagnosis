# gradcampp.py
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_gradcampp(model, image, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, axis=0))
        pred_index = tf.argmax(predictions[0])
        output = predictions[:, pred_index]
        
    grads = tape1.gradient(output, conv_outputs)
    grads2 = tape2.gradient(output, conv_outputs)
    grads3 = tape3.gradient(output, conv_outputs)
    
    numerator = grads2[0] ** 2
    denominator = 2 * grads2[0] + tf.reduce_sum(conv_outputs[0] * grads3[0], axis=(0, 1)) + 1e-8
    alpha = numerator / denominator

    weights = tf.reduce_sum(alpha * tf.nn.relu(grads[0]), axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / tf.reduce_max(cam)
    cam = tf.image.resize(cam[..., tf.newaxis], (224, 224))
    return cam.numpy().squeeze()

def overlay_heatmap(image, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    image = np.uint8(255 * image)
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlayed
