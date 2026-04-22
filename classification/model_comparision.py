import matplotlib.pyplot as plt
import pandas as pd

# ==============================
# YOUR ACTUAL RESULTS
# ==============================
mobilenet_acc = 885 / 1503   # ≈ 0.589
resnet_acc = 1112 / 1503     # ≈ 0.740

data = {
    "Model": ["MobileNetV2", "ResNet18"],
    "Accuracy": [mobilenet_acc, resnet_acc]
}

df = pd.DataFrame(data)

# ==============================
# PLOT
# ==============================
plt.figure(figsize=(6,4))
plt.bar(df["Model"], df["Accuracy"])
plt.title("Model Accuracy Comparison (HAM10000)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("outputs/model_accuracy_comparison.png")
plt.show()

print("✅ Saved: outputs/model_accuracy_comparison.png")