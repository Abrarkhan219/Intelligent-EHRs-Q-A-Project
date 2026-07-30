import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("report/figures", exist_ok=True)

def make_distribution_figures(csv_path, prefix, label):
    df = pd.read_csv(csv_path)
    for metric in ["precision", "recall", "f1", "bleu"]:
        plt.figure(figsize=(7, 4))
        plt.hist(df[metric], bins=30, edgecolor="black")
        plt.title(f"{metric.upper()} Distribution — {label} (n={len(df)})")
        plt.xlabel(f"{metric} score")
        plt.ylabel("Number of questions")
        plt.tight_layout()
        plt.savefig(f"report/figures/{prefix}_{metric}_distribution.png", dpi=150)
        plt.close()
    print(f"Saved 4 figures for: {label}")
    return df[["precision", "recall", "f1", "bleu"]].mean()

end_to_end_means = make_distribution_figures(
    "results/evaluation_results.csv", "e2e", "End-to-End Pipeline"
)
oracle_means = make_distribution_figures(
    "results/evaluation_oracle_results.csv", "oracle", "Oracle-Context Generation"
)

# Comparison bar chart: end-to-end vs oracle, side by side
comparison = pd.DataFrame({"End-to-End": end_to_end_means, "Oracle-Context": oracle_means})
comparison.plot(kind="bar", figsize=(8, 5))
plt.title("End-to-End vs. Oracle-Context: Mean Metric Comparison")
plt.ylabel("Mean score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("report/figures/comparison_e2e_vs_oracle.png", dpi=150)
plt.close()
print("Saved comparison_e2e_vs_oracle.png")

print("\nEnd-to-end means:\n", end_to_end_means)
print("\nOracle means:\n", oracle_means)