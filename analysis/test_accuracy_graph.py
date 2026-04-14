import matplotlib.pyplot as plt

models = ['ResNet50', 'EfficientNetB0', 'DenseNet121', 'MobileNetV2']
accuracy = [93.63, 79.41, 89.48, 58.52]

plt.figure(figsize=(8,5))

bars = plt.bar(models, accuracy)

plt.ylabel("Accuracy (%)")
plt.xlabel("Models")
plt.title("Model Accuracy Comparison")

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.5, f'{y:.2f}%', ha='center')

plt.ylim(0,100)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()