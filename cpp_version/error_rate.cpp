


struct ConfusionMatrix {
  int32_t num_categories;
  int32_t num_samples;
  std::vector<int32_t> tp; // True Positive
  std::vector<int32_t> fp; // False Positive
  std::vector<int32_t> fn; // False Negative
};

ConfusionMatrix create_confusion_matrix(int32_t num_categories) {
  std::vector<int32_t> tp(num_categories);
  std::vector<int32_t> fp(num_categories);
  std::vector<int32_t> fn(num_categories);

  return ConfusionMatrix{
    .num_categories = num_categories,
    .num_samples = 0,
    .tp = std::move(tp),
    .fp = std::move(fp),
    .fn = std::move(fn),
  };
}

// Record all category outputs in one run.
void record(ConfusionMatrix& matrix, std::vector<int32_t> true_labels, std::vector<int32_t> predicted_labels) {
  int32_t n = matrix.num_categories;

  for (int i = 0; i < n; i++) {
    if ((true_labels[i] == 1) && (predicted_labels[i] == 1)) {
      matrix.tp[i]++; // True Positive: Prediction == 1 && Prediction == Truth
    }
    else if ((true_labels[i] == 0) && (predicted_labels[i] == 1)) {
      matrix.fp[i]++; // False Positive: Prediction == 1 && Prediction != Truth
    }
    else if ((true_labels[i] == 1) && (predicted_labels[i] == 0)) {
      matrix.fn[i]++; // False Negative: Prediction == 0 && Prediction != Truth
    }
  }
  return;
}

float error_rate(const ConfusionMatrix& matrix) {
  int32_t n = matrix.num_categories;
  int32_t samples = matrix.num_samples;
  if (samples == 0) {
    return 0;
  }

  int32_t total_errors = 0;
  for (const auto& signal : matrix.fp) {
      total_errors += signal;
  }
  for (const auto& signal : matrix.fn) {
      total_errors += signal;
  }

  int32_t total_decisions = n * samples;
  return total_errors / total_decisions;
}

// Calculate the count of nodes with state 1 in the manifold.
int32_t calcualte_active_nodoes() {
  int32_t active_state_count = 0;
  for (int32_t word : global_array) {
    if (word & 1) {
      active_state_count++;
    }
  }
  return active_state_count;
}

// F1-score calculation is put in the summary.
void summary(const ConfusionMatrix& matrix) {
  printf("*** Active node count: {%d} ***\n", calcualte_active_nodoes());

  int32_t samples = matrix.num_samples;
  float rate = error_rate(matrix);
  printf("Samples: {%d}, Error rate: {%f:.4f}\n", samples, rate);

  // F1-score calculation
  int32_t n = matrix.num_categories;
  std::vector<float> per_class(n);
  int32_t count = 0;
  float macro;
  {
    for (int i = 0; i < n; i++) {
      // If all TP, FP and FN are 0, then skip calculating f1_score for it since it's not useful in the calculation.
      if (matrix.tp[i] + matrix.fp[i] + matrix.fn[i] == 0) {
        continue;
      }

      // Precision = TP / (TP + FP)
      // Recall = TP / (TP + FN)
      // F1 = 2 * Precision * Recall / (Precision + Recall)
      int32_t denom = 2 * matrix.tp[i] + matrix.fp[i] + matrix.fn[i];
      per_class[i] = denom == 0 ? 0 : (2 * matrix.tp[i] / denom);
      count++;
    }

    float sum = 0;
    for (const auto& micro : per_class) {
        sum += micro;
    }

    macro = sum / count;
  }
  printf("Count in F1-score cal: %d\n", count);
  printf("Macro F1-Score: {%f:.4f}\n", macro);
  // if per_class:
  //     parts = [f"cat {c}: {f1:.4f}" for c, f1 in sorted(per_class.items())]
  //     print("Per-category F1: " + ", ".join(parts))
}

// A summary of the shape of manifold/global_array before the execution.
void pre_run_summary() {
  printf("=======\n PreRun Active node count: {%d} \n=======\n", calcualte_active_nodoes());
}