import pandas as pd
import re

df = pd.read_csv("results/evaluation_results.csv")

# Group by the fixed question templates used in build_validation_set.py
summary = df.groupby("question")[["precision", "recall", "f1", "bleu"]].agg(["mean", "count"])
print(summary)

summary.to_csv("results/breakdown_by_question_type.csv")
print("\nSaved: results/breakdown_by_question_type.csv")