import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([
[131,0,0,0,4],
[0,135,0,0,0],
[0,0,127,4,4],
[0,0,13,120,2],
[4,0,12,0,119]
])

classes = ['breaker','green','pink','red','turning']

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - ResNet50")

plt.tight_layout()
plt.savefig("resnet_confusion_matrix.png", dpi=300)
plt.show()