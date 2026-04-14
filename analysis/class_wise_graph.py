import matplotlib.pyplot as plt
import numpy as np

classes = ['breaker', 'green', 'pink', 'red', 'turning']

# F1 scores from your classification reports
resnet50 = [0.97, 1.00, 0.89, 0.93, 0.90]
efficientnet = [0.84, 0.96, 0.63, 0.83, 0.71]
densenet = [0.91, 0.99, 0.82, 0.93, 0.83]
mobilenet = [0.66, 0.47, 0.52, 0.64, 0.58]

x = np.arange(len(classes))
width = 0.2

plt.figure(figsize=(10,6))

plt.bar(x - 1.5*width, resnet50, width, label='ResNet50')
plt.bar(x - 0.5*width, efficientnet, width, label='EfficientNetB0')
plt.bar(x + 0.5*width, densenet, width, label='DenseNet121')
plt.bar(x + 1.5*width, mobilenet, width, label='MobileNetV2')

plt.xticks(x, classes)
plt.ylabel('F1 Score')
plt.xlabel('Classes')
plt.title('Class-wise F1 Score Comparison of Deep Learning Models')
plt.ylim(0,1.1)
plt.legend()

plt.tight_layout()
plt.savefig("classwise_f1_comparison.png", dpi=300)
plt.show()