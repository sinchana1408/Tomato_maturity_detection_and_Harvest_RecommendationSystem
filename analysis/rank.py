import matplotlib.pyplot as plt
import numpy as np

models = ['ResNet50', 'EfficientNetB0', 'DenseNet121', 'MobileNetV2']

accuracy = [93.63, 79.41, 89.48, 58.52]
precision = [93.92, 80.58, 89.65, 70.75]
recall = [93.63, 79.41, 89.48, 58.52]
f1 = [93.67, 79.54, 89.46, 57.29]

# Calculate average performance
avg_score = np.mean([accuracy, precision, recall, f1], axis=0)

plt.figure(figsize=(8,5))

bars = plt.bar(models, avg_score)

plt.ylabel("Average Performance Score (%)")
plt.xlabel("Models")
plt.title("Overall Model Performance Ranking")

# Show score on top of bars
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.5, f'{y:.2f}', ha='center')

plt.ylim(0,100)

plt.tight_layout()
plt.savefig("model_performance_ranking.png", dpi=300)
plt.show()