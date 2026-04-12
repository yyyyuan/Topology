### Use confusion matrix and F1 score to do calculation.

# Output precous rate calculation
def create_confusion_matrix(num_categories=1000):
    return {
        "num_categories": num_categories,
        "num_samples": 0,
        "tp": [0] * num_categories,
        "fp": [0] * num_categories,
        "fn": [0] * num_categories,
    }

# Record all category outputs in one run.
def record(matrix, true_labels, predicted_labels):
    n = matrix["num_categories"]
    matrix["num_samples"] += 1
    for i in range(n):
        t = true_labels[i]
        p = predicted_labels[i]
        if t == 1 and p == 1:
            matrix["tp"][i] += 1 # True Positive: Prediction == 1 && Prediction == Truth
        elif t == 0 and p == 1:
            matrix["fp"][i] += 1 # False Positive: Prediction == 1 && Prediction != Truth
        elif t == 1 and p == 0:
            matrix["fn"][i] += 1 # False Negative: Prediction == 0 && Prediction != Truth


def error_rate(matrix):
    n = matrix["num_categories"]
    samples = matrix["num_samples"]
    if samples == 0:
        return 0.0
    total_errors = sum(matrix["fp"]) + sum(matrix["fn"])
    total_decisions = n * samples
    return total_errors / total_decisions


def f1_score(matrix):
    n = matrix["num_categories"]
    per_class = {}
    for i in range(n):
        tp = matrix["tp"][i]
        fp = matrix["fp"][i]
        fn = matrix["fn"][i]

        # If all TP, FP and FN are 0, then skip calculating f1_score for it since it's not useful in the calculation.
        if tp + fp + fn == 0:
            continue

        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN)
        # F1 = 2 * Precision * Recall / (Precision + Recall)
        denom = 2 * tp + fp + fn
        per_class[i] = (2 * tp / denom) if denom else 0.0

    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return per_class, macro


def summary(matrix):
    samples = matrix["num_samples"]
    rate = error_rate(matrix)
    print(f"Samples: {samples}, Error rate: {rate:.4f}")

    per_class, macro = f1_score(matrix)
    print(f"Macro F1-Score: {macro:.4f}")
    if per_class:
        parts = [f"cat {c}: {f1:.4f}" for c, f1 in sorted(per_class.items())]
        print("Per-category F1: " + ", ".join(parts))
