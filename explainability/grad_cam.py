import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model("ResNet50.keras")

# Last convolution layer name (important!)
LAST_CONV_LAYER = "conv5_block3_out"   # for ResNet50

# Load and preprocess image
def load_image(img_path, size=(224,224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    return img_array

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Superimpose heatmap on original image
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * alpha + img

    plt.imshow(cv2.cvtColor(superimposed.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# =========================
# Run Grad-CAM
# =========================

img_path = r"E:\MAJOR_PROJECT\cropped_original_model\dataset\test\turner\turner_00085.jpg"   # give any test tomato image

img_array = load_image(img_path)
heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)

overlay_heatmap(img_path, heatmap)