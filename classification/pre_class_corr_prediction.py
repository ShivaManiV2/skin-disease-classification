import matplotlib.pyplot as plt
import numpy as np

# ==============================
# DIAGONAL VALUES FROM MATRICES
# ==============================
mobilenet_correct = [36, 44, 104, 4, 72, 607, 18]
resnet_correct = [27, 54, 104, 11, 101, 797, 18]

classes = ["bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(10,5))
plt.bar(x - width/2, mobilenet_correct, width, label="MobileNetV2")
plt.bar(x + width/2, resnet_correct, width, label="ResNet18")

plt.xlabel("Skin Disease Class")
plt.ylabel("Correct Predictions")
plt.title("Per-Class Performance Comparison")
plt.xticks(x, classes, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("outputs/per_class_comparison.png")
plt.show()

print("✅ Saved: outputs/per_class_comparison.png")