import numpy as np
import matplotlib.pyplot as plt

# Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Model scores
resnet = [93.63, 93.92, 93.63, 93.67]
efficientnet = [79.41, 80.58, 79.41, 79.54]
densenet = [89.48, 89.65, 89.48, 89.46]
mobilenet = [58.52, 70.75, 58.52, 57.29]

# Convert to numpy
labels = np.array(metrics)
num_vars = len(labels)

angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

# Close the plots
resnet = np.concatenate((resnet, [resnet[0]]))
efficientnet = np.concatenate((efficientnet, [efficientnet[0]]))
densenet = np.concatenate((densenet, [densenet[0]]))
mobilenet = np.concatenate((mobilenet, [mobilenet[0]]))

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, polar=True)

ax.plot(angles, resnet, label='ResNet50')
ax.fill(angles, resnet, alpha=0.1)

ax.plot(angles, efficientnet, label='EfficientNetB0')
ax.fill(angles, efficientnet, alpha=0.1)

ax.plot(angles, densenet, label='DenseNet121')
ax.fill(angles, densenet, alpha=0.1)

ax.plot(angles, mobilenet, label='MobileNetV2')
ax.fill(angles, mobilenet, alpha=0.1)

ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)

plt.title("Performance Comparison of Deep Learning Models")
plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))

plt.savefig("model_radar_chart.png", dpi=300)
plt.show()