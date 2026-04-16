#include <iostream>
#include <math.h>
#include <cstdint>  // Required for int32_t
#include <vector>


// ============ Constants ============
int32_t ADDR_BITS = 22;
int32_t K_DIM = 22;  // hypercube dimension for logical addresses

int32_t SHIFT_ADDR = 1 + 1 + 5 + 3;
int32_t SHIFT_COUNTER = 1 + 1 + 5;
int32_t SHIFT_K = 1 + 1;
int32_t SHIFT_DIR = 1;
int32_t SHIFT_STATE = 0;

int32_t MASK_ADDR = (1 << (ADDR_BITS)) - 1;
int32_t MASK_COUNTER = 0b111;
int32_t MASK_K = 0b11111;
int32_t MASK_DIR = 0b1;
int32_t MASK_STATE = 0b1;

int32_t U32_MASK = 0xFFFFFFFF;
// Only bits related to k are set to 0, 0b11111111111111111111111110000011 <- 32-bits
int32_t K_MASK_IN_WORD = 0xFFFFFF83;
// Only bits related to directions are set to 0, 0b11111111111111111111111111111101 <- 32-bits
int32_t DIRECTION_MASK_IN_WORD = 0xFFFFFFFD;

// Bit 30: 1 = increase k when strength hits 0; 0 = decrease k (with wrap in 5 bits).
int32_t DIR_INCREASE_K = 1;
int32_t DIR_DECREASE_K = 0;

// ========= Parameters related to I/O (image read) =========
int32_t IMAGE_WIDTH = 64;
int32_t IMAGE_HEIGHT = 64;
int32_t CHANNELS = 3; // RGB has 3 channcels in each pixel.
int32_t BITS_PER_CHANNEL = 8;

int32_t INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS * BITS_PER_CHANNEL;  // 98304 = 64 * 64 * 3 * 8 bits for one image.
int32_t OUTPUT_SIZE = 1000; // Let's do 1000 different categories.

std::string IMAGE_PATH = "test_image.png";

int32_t IDX_DIMENSION_0_RANGE = IMAGE_HEIGHT * IMAGE_WIDTH / 2;
int32_t IDX_DIMENSION_1_RANGE = (1 << (ADDR_BITS - 1)) + IDX_DIMENSION_0_RANGE; // 2^(ADDR_BITS - 1) + Range

// The range of Reaction must be larger than 2024 and out of input dimension range.
// The maximum range means there is very long history of correct predictions.
int32_t IDX_REACTION_RANGE_START = 3000;
int32_t IDX_REACTION_RANGE_END = 4000;

// =========== End of Constants =========

// ========== Global array ======
std::vector<int32_t> global_array(1 << ADDR_BITS);

// ========== Operators related to words =========
int32_t pack_word(int32_t addr, int32_t counter, int32_t k, int32_t dir_bit, int32_t state) {
  return (state & MASK_STATE)
        | ((dir_bit & MASK_DIR) << SHIFT_DIR)
        | ((k & MASK_K) << SHIFT_K)
        | ((counter & MASK_COUNTER) << SHIFT_COUNTER)
        | ((addr & MASK_ADDR) << SHIFT_ADDR);
}

int32_t unpack_addr(int32_t word) {
    return (word >> SHIFT_ADDR) & MASK_ADDR;
}

int32_t unpack_counter(int32_t word) {
    return (word >> SHIFT_COUNTER) & MASK_COUNTER;
}

int32_t unpack_k(int32_t word) {
    return (word >> SHIFT_K) & MASK_K;
}

int32_t unpack_dir(int32_t word) {
    return (word >> SHIFT_DIR) & 1;
}

int32_t unpack_state(int32_t word) {
    return word & MASK_STATE;
}

int32_t with_addr(int32_t word, int32_t logical_addr) {
    return (word & ~(MASK_ADDR << SHIFT_ADDR)) | ((logical_addr & MASK_ADDR) << SHIFT_ADDR);
}

int32_t with_strength(int32_t word, int32_t strength) {
    return (word & ~(MASK_COUNTER << SHIFT_COUNTER)) | (
        (strength & MASK_COUNTER) << SHIFT_COUNTER
    );
}

int32_t with_k(int32_t word, int32_t k) {
    return (word & ~(MASK_K << SHIFT_K)) | ((k & MASK_K) << SHIFT_K);
}

int32_t with_direction(int32_t word, int32_t direction) {
    return (word & ~(MASK_DIR << SHIFT_DIR)) | (
        (direction & MASK_DIR) << SHIFT_DIR
    );
}

int32_t with_state(int32_t word, int32_t state) {
    return (word & ~(MASK_STATE << SHIFT_STATE)) | (
        (state & MASK_STATE) << SHIFT_STATE
    );
}

// 1 if 1-bit states differ, else 0.
int32_t resonates(int32_t self_state, int32_t neighbor_state) {
    return (self_state ^ neighbor_state) & 1;
}

void decide_k_and_dirction(int32_t& k, int32_t& direction) {
    if (k == 0 && direction != DIR_INCREASE_K) {
        direction = DIR_INCREASE_K;
    }
    if (k == ADDR_BITS && direction == DIR_INCREASE_K) {
        direction = DIR_DECREASE_K;
    }

    if (direction == DIR_INCREASE_K) {
        k = (k + 1) & MASK_K;
    }
    else {
        k = (k - 1) & MASK_K;
    }

    return;
}

int32_t update_k_and_direction_within_word(int32_t word) {
    int32_t direction = unpack_dir(word);
    int32_t k = unpack_k(word);

    decide_k_and_dirction(k, direction);

    return word & K_MASK_IN_WORD & DIRECTION_MASK_IN_WORD | ((k & MASK_K) << SHIFT_K) | ((direction & MASK_DIR) << SHIFT_DIR);
}

bool is_input_range(int32_t index) {
  if ((index >= 0 && index < IDX_DIMENSION_0_RANGE) ||
      (index >= (1 << (ADDR_BITS - 1)) && index < IDX_DIMENSION_1_RANGE)) {
      return true;
  }
  return false;
}

bool is_output_range(int32_t index) {
  if (index >= ((1 << ADDR_BITS) - OUTPUT_SIZE) && (index < (1 << ADDR_BITS))) {
    return true;
  }
  return false;
}

bool is_reaction_range(int32_t index) {
  if (index >= IDX_REACTION_RANGE_START && index < IDX_REACTION_RANGE_END) {
    return true;
  }

  return false;
}

// The heartbeat function is the core of algorithm.
int32_t heartbeat(int32_t w) {
  int32_t address = unpack_addr(w);
  int32_t direction = unpack_dir(w);
  int32_t k = unpack_k(w);
  int32_t strength = unpack_counter(w);
  int32_t self_state = unpack_state(w);

  k = std::min(k, ADDR_BITS - 1); // In current implementation, the maximum value of k is 21.

  int32_t neighbor_index = address ^ (1 << k); // Flip the bit at k
  int32_t neighbor = global_array.at(neighbor_index);

  // Skip heartbeat over reaction nodes since they will be taken care of separately.
  // if (is_reaction_range(address)) {
  //   return w;
  // }

  // Output node cannot be used as source to pull data from.
  // Instead we update sections k and direction inside the word directly.
  if (is_output_range(neighbor_index)) {
    return update_k_and_direction_within_word(w);
  }

  int32_t neighbor_state = unpack_state(neighbor);
  bool resonate = self_state != neighbor_state;

  if (resonate) {
      strength = std::min(7, strength + 1);
      self_state = neighbor_state;
  }
  else {
      strength = std::max(0, strength - 1);
  }

  // Change to next index if the counter strength is 0.
  if (strength == 0) {
      decide_k_and_dirction(k, direction);
  }

  return pack_word(address, strength, k, direction, self_state);
}
// =============== End of Word Related Operators ==========

// ============ Image loading ===============
// This function currently only loads one image into manifold.
// Hence signals in input nodes don't need to change.
// TODO: Upgrade this function so it can read tons of data from sources such as ImageNet.
void load_image_to_manifold(int32_t input_range) {
  // Hardcode the image in path "/content/test_image.png" into manifold to simplify the POC.
  int32_t idx_dimension0 = 0;
  int32_t idx_dimension1 = global_array.size() / 2; // Idx must be integer
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    for (int x = 0; x < IMAGE_WIDTH; x++) {
      if ((idx_dimension0 + idx_dimension1 - (global_array.size() / 2)) > input_range) {
        break;
      }
      float t = (x + y) / 126.0;
      int32_t r = int(255 * (1 - t));
      int32_t g = int(255 * std::min(t, 1 - t) * 2);
      int32_t b = int(255 * t);

      bool is_dimension_0 = x < IMAGE_WIDTH / 2;

      for (int bit_pos = 7; bit_pos >= 0; bit_pos--) {
        if (is_dimension_0) {
          global_array[idx_dimension0] = pack_word(idx_dimension0, 1, 0, 1, (r >> bit_pos) & 1);
          idx_dimension0++;
        } else {
          global_array[idx_dimension1] = pack_word(idx_dimension1, 1, 0, 1, (r >> bit_pos) & 1);
          idx_dimension1++;
        }
      }

      for (int bit_pos = 7; bit_pos >= 0; bit_pos--) {
        if (is_dimension_0) {
          global_array[idx_dimension0] = pack_word(idx_dimension0, 1, 0, 1, (g >> bit_pos) & 1);
          idx_dimension0++;
        } else {
          global_array[idx_dimension1] = pack_word(idx_dimension1, 1, 0, 1, (g >> bit_pos) & 1);
          idx_dimension1++;
        }
      }

      for (int bit_pos = 7; bit_pos >= 0; bit_pos--) {
        if (is_dimension_0) {
          global_array[idx_dimension0] = pack_word(idx_dimension0, 1, 0, 1, (b >> bit_pos) & 1);
          idx_dimension0++;
        } else {
          global_array[idx_dimension1] = pack_word(idx_dimension1, 1, 0, 1, (b >> bit_pos) & 1);
          idx_dimension1++;
        }
      }
    }
  }
  return;
}
// ========== End of Image Loading =============

// ========== Error Rate Calcualtions ==========

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

// F1-score calculation is put in the summary.
void summary(const ConfusionMatrix& matrix) {
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
      if (matrix.tp[i] + matrix.fp[0] + matrix.fn[0] == 0) {
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
  printf("Macro F1-Score: {%f:.4f}\n", macro);
  // if per_class:
  //     parts = [f"cat {c}: {f1:.4f}" for c, f1 in sorted(per_class.items())]
  //     print("Per-category F1: " + ", ".join(parts))
}
// ========== End of Error Rate Calculations ===


int main(void)
{
  // Init the global_array.
  for (int i = 0; i < (1 << ADDR_BITS); i++) {
    global_array[i] = pack_word(i, 1, 0, 1, 0);
  }
  load_image_to_manifold(INPUT_SIZE);

  int32_t n = 1 << ADDR_BITS;

  ConfusionMatrix cm = create_confusion_matrix(OUTPUT_SIZE);
  std::vector<int32_t> true_labels(OUTPUT_SIZE);
  std::vector<int32_t> pred_labels(OUTPUT_SIZE);

  int32_t is_output_category_matching = 0;
  int32_t predictions_made = 0;
  std::vector<int32_t> output_category_correctness(IDX_REACTION_RANGE_END - IDX_REACTION_RANGE_START);

  // For one single image, there is only 1 matching category out of all 1000 categories.
  // TODO: Eventually this should be modified by datasets, instead of being hardcoded.
  true_labels[427] = 1;
  std::vector<int32_t> prediction_signals(OUTPUT_SIZE);
  int32_t init_prediction_sngal = 0; // The signal used to indicate if this is a good prediction.
  int32_t init_reaction_signal = 1;  // The reaction signal stream is totally flipped in each heartbeat.

  // TODO: The manifold is able to accept arbitrary signals other than standard image inputs.
  //       Those arbitrary signals are vital to evolve the manifold to generate expected outputs.
  //       It's easy to see such architecture is extendable, to meet all different types of real-world (physical) requirements.

  while (true) {
    for (int index = 0; index < n; index++) {
      if (is_input_range(index)) {
        // Increment over input nodes, which don't pull signals from other nodes.
        continue;
      }

      // The core calculation.
      int32_t updated_word = heartbeat(global_array[index]);
      global_array[index] = updated_word;

      if (is_output_range(index)) {
        int32_t pred_category = unpack_state(updated_word);

        // The offset of prediction array is (2**ADDR_BITS - OUTPUT_SIZE)
        int32_t offset = index - ((1 << ADDR_BITS) - OUTPUT_SIZE);
        if (pred_category == 1) {
          predictions_made++;
        }

        int32_t prediction_made = pred_category;
        if (prediction_signals[offset] != pred_category) {
          prediction_made = 1;
          prediction_signals[offset] = 1 - prediction_signals[offset];
        }

        int32_t count_of_wrong_guesses = 0;
        if (offset == 427 && pred_category == 1) {
          printf("offset 247: %d\n", pred_category);
          output_category_correctness[offset] += 10;
          is_output_category_matching += output_category_correctness[offset];
        }
        else if ((offset == 427 && pred_category == 0)) {
          // output_category_correctness[offset]--;      
          printf("offset 247: %d\n", pred_category);
          output_category_correctness[offset]--;
          output_category_correctness[offset] = std::max(0, output_category_correctness[offset]);
          is_output_category_matching += output_category_correctness[offset];
          count_of_wrong_guesses++;  
        }
        else if (offset != 427 && pred_category == 1) {
          is_output_category_matching--;
          is_output_category_matching = std::max(0, is_output_category_matching);
          count_of_wrong_guesses++;  
        }

        // A constant 1->0->1 or 0->1->1 represent the prediction.
        pred_labels[offset] = pred_category;
        cm.num_samples++;
      }
      // if (index % 100000 == 0) {
      //   printf("index: %d \n", index);
      // }
    }

    // After each round of calculation, decide the scale of correct prediction stimulation.
    // int32_t reaction_signal = init_reaction_signal;
    // for (int i = 0; i < (IDX_REACTION_RANGE_END - IDX_REACTION_RANGE_START); i++) {
    //   if (i < is_output_category_matching) {
    //     global_array[i + IDX_REACTION_RANGE_START] = pack_word(i + IDX_REACTION_RANGE_START, 1, 0, 1, reaction_signal);
    //     reaction_signal = 1 - reaction_signal;
    //   }
    // }
    // init_reaction_signal = 1 - init_reaction_signal;
    load_image_to_manifold(is_output_category_matching);  // Control the input range after evaluation.
    printf("predictions made: %d\n", predictions_made);
    printf("next round input range: %d\n", is_output_category_matching);
    predictions_made = 0;  // Reset the predictions_made.
    is_output_category_matching = 0;  // Reset the output evaluation.

    // After one full round calculation of all nodes in the manifold, calculate the F1-score on error rates and reset the confusion matrix.
    // Run the record of all category outputs in one run.
    record(cm, true_labels, pred_labels);
    // printf("Recorded {pred_labels.count(1)} predictions with 1.");
    summary(cm);
    pred_labels.assign(OUTPUT_SIZE, 0);
    cm = create_confusion_matrix(OUTPUT_SIZE);
  // TODO: Migrate the function into GPU and see if magic happens!
  }
}