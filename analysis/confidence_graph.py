# Import required libraries
import matplotlib.pyplot as plt
import numpy as np

# Image names
images = ["breaker", "green", "pink", "red", "turning"]

# Confidence values for each model
mobilenet = [93.02, 99.99, 88.22, 95.54, 69.30]
densenet = [88.81, 99.50, 89.79, 97.31, 90.76]
efficientnet = [88.81, 99.50, 89.79, 97.31, 90.76]
resnet = [99.90, 100.00, 99.94, 99.98, 99.93]

# X-axis positions
x = np.arange(len(images))
width = 0.2

# Create figure
plt.figure(figsize=(12, 6))

# Plot grouped bars
bars1 = plt.bar(x - 1.5*width, mobilenet, width, label="MobileNet")
bars2 = plt.bar(x - 0.5*width, densenet, width, label="DenseNet121")
bars3 = plt.bar(x + 0.5*width, efficientnet, width, label="EfficientNetB0")
bars4 = plt.bar(x + 1.5*width, resnet, width, label="ResNet50")

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, 
                 height + 0.5, 
                 f'{height:.1f}', 
                 ha='center', 
                 va='bottom', 
                 fontsize=8)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Labels and title
plt.xlabel("Test Images", fontsize=12)
plt.ylabel("Confidence (%)", fontsize=12)
plt.title("Prediction Confidence Comparison Per Image", fontsize=14)
plt.xticks(x, images)
plt.ylim(0, 105)

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()