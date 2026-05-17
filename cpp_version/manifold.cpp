#include <iostream>
#include <math.h>
#include <cstdint>  // Required for int32_t
#include <vector>
#include <algorithm> // For std::shuffle
#include <random>    // For std::mt19937 and std::random_device
#include <numeric>   // For std::iota
#include <string>
#include <fstream>
#include <execution> // Required for parallel policies
#include <omp.h> // Header for OpenMP functions
#include <atomic>

#include "constants.h"
#include "database.h"
#include "manifold_operators.h"

/*
Format:
--------------------------------------
| addr | counter |   k   | dir| state|
| 0-21 |  22-24  | 25-29 | 30 |  31  |
--------------------------------------

For each value in integer format, there are 16 different changes it could be.
k has 5 bits, in theory its maximum value is 31.
*/

// ========== Operators related to words =========

// =============== End of Word Related Operators ==========

// ============ Image loading ===============
// This function currently only loads one image into manifold.
// Hence signals in input nodes don't need to change.
// TODO: Upgrade this function so it can read tons of data from sources such as ImageNet.
// TODO: Change the input formats from constant 1s into 0<->1 pulse to mimic input signals.
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
  IDX_DIMENSION_0_RANGE = idx_dimension0 + 1;
  IDX_DIMENSION_1_RANGE = idx_dimension1 + 1;
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

// ========== End of Error Rate Calculations ===

// ========== Index Array Randomization =======

void shuffle() {
  // Use random_device to seed the generator
  std::random_device rd;
  std::mt19937 g(rd());

  // Shuffle the index array
  std::shuffle(index_array.begin(), index_array.end(), g);
}

// Randomly select a value between 0 and 21. Used in global_array initialization.
// Currently the maximum value of k should be 21, to align with the hybercube dimension (22).
uint8_t random_generate_k() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 21);
    return static_cast<uint8_t>(dist(rng));
}

// Randomly select a direction of the next k change. Used in global_array initialization.
uint8_t random_generate_direction() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 1);
    return static_cast<uint8_t>(dist(rng));
}

// ========== End of Index Array Randomization =======

// ========== Manifold Loading & Saving =========
bool save_manifold(const std::vector<int32_t>& manifold_array, const std::string& filepath, const std::string& debug_filepath) {
    std::ofstream out(filepath, std::ios::out | std::ios::trunc);
    std::ofstream debug(debug_filepath, std::ios::out | std::ios::trunc);
    if (!out.is_open() || !debug.is_open()) return false;
    for (int32_t n : manifold_array) {
        out << n << '\n';
        debug << std::bitset<32>(static_cast<uint32_t>(n)) << '\n';
    }
    return out.good();
}

bool load_manifold(std::vector<int32_t>& manifold_array, const std::string& filepath) {
    std::vector<int32_t> input_array;
    std::ifstream in(filepath);
    if (!in.is_open()) return false;
    double value;
    while (in >> value) {
        input_array.push_back(value);
    }

    if (input_array.size() != (1 << ADDR_BITS)) {
      return false;
    }

    manifold_array = std::move(input_array);
    return true;
}

// ========== End of Manifold Loading & Saving ============

int main(int argc, char* argv[])
{
  bool use_save_data = false;
  int32_t maximum_runs = 10;
  bool need_init = false;
  for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--use_save_data") {
        use_save_data = true;
      }
      if (arg == "--runs" && i + 1 < argc) {
        maximum_runs = static_cast<int32_t>(std::atoi(argv[++i]));
      }
      if (arg == "--init") {
        need_init = true;
      }
  }

  // Init the global_array.
  if (use_save_data) {
    need_init = !load_manifold(global_array, MANIFOLD_PATH);
  }
  printf("Do we need to re-init manifold array: %d\n", need_init);

  // Initialize the global_array
  for (int i = 0; i < (1 << ADDR_BITS); i++) {
    if (need_init) {
      global_array[i] = pack_word(i, 8, random_generate_k(), random_generate_direction(), 1);
    }
    cycle_delays_array[i] = 7;  // Start with maximum cycle delay when the connection strength is 0.
    index_array[i] = i;
  }
  load_image_to_manifold(INPUT_SIZE);

  int32_t n = 1 << ADDR_BITS;

  ConfusionMatrix cm = create_confusion_matrix(OUTPUT_SIZE);
  std::vector<int32_t> true_labels(OUTPUT_SIZE);
  std::vector<int32_t> pred_labels(OUTPUT_SIZE);

  std::atomic<int32_t> is_output_category_matching = 0;
  std::atomic<int32_t> predictions_made = 0;
  std::vector<int32_t> output_category_correctness(IDX_REACTION_RANGE_END - IDX_REACTION_RANGE_START);
  bool is_prediction_hit = false; // True if the correct prediction is made.

  // For one single image, there is only 1 matching category out of all 1000 categories.
  // TODO: Eventually this should be modified by datasets, instead of being hardcoded.
  true_labels[427] = 1;
  std::vector<int32_t> prediction_signals(OUTPUT_SIZE);
  int32_t init_prediction_sngal = 0; // The signal used to indicate if this is a good prediction.
  int32_t init_reaction_signal = 1;  // The reaction signal stream is totally flipped in each heartbeat.

  // TODO: The manifold is able to accept arbitrary signals other than standard image inputs.
  //       Those arbitrary signals are vital to evolve the manifold to generate expected outputs.
  //       It's easy to see such architecture is extendable, to meet all different types of real-world (physical) requirements.
  // TODO: Special handlings to achieve "light speed limitation" in both CPU and GPU infra.
  //       Currently this manifold runs on CPU and is naturally running in synchronization,
  //       we need to build an extra array storing the cycle deplay for each connetion.
  //       In GPU mode, the manifold runs asynchronatically so the cycle delay is achived from asynchronizatin naturally.
  // TODO: Add a new constant 0/1 pulse into manifold and see if it affects the manifold evolvement.
  // TODO: Add a non-linearity into node connection, after the node state becomes 1, it will flip back to 0 again in the next run.
  //       This behavior only exists in nodeds in manifold, allowing inputs to be a constant stream of 1s if necessary.
  //       The output nodes still follow non-linearity behavior, so it outputs are still a constant flip of 0/1s.

  // shuffle();  // Do a shuffle before the execution!
  int count = 0;
  pre_run_summary();
  while (count++ < maximum_runs) {
    #pragma omp parallel for
    for (int index = 0; index < n; index++) {
      if (index == 0) {
        std::cout << "Running with " << omp_get_num_threads() << " threads." << std::endl;
      }
      int32_t global_array_index = index_array[index];
      if (is_input_range(global_array_index)) {
        // Increment over input nodes, which don't pull signals from other nodes.
        continue;
      }

      // The core calculation.
      int32_t current_word = global_array[global_array_index];
      int32_t updated_word = heartbeat(current_word);
      global_array[global_array_index] = updated_word;

      if (is_output_range(global_array_index)) {
        int32_t previous_prediction = unpack_state(current_word);
        int32_t pred_category = unpack_state(updated_word);
        int32_t counter = unpack_counter(updated_word);
        // Only 1->0 is recognized as a fire signal.
        // bool fire_output_signal = pred_category != previous_prediction;
        bool fire_output_signal = counter >= 4;  // The connection strength represents the output signal.

        // The offset of prediction array is (2**ADDR_BITS - OUTPUT_SIZE)
        int32_t offset = global_array_index - ((1 << ADDR_BITS) - OUTPUT_SIZE);
        if (fire_output_signal) {
          predictions_made++;
        }

        int32_t prediction_made = pred_category;
        if (prediction_signals[offset] != pred_category) {
          prediction_made = 1;
          prediction_signals[offset] = 1 - prediction_signals[offset];
        }

        int32_t count_of_wrong_guesses = 0;
        if (offset == 427 && fire_output_signal) {
          printf("offset 247: %d\n", pred_category);
          output_category_correctness[offset] += 1;
          // is_output_category_matching += output_category_correctness[offset];
          is_output_category_matching += 5;
          is_prediction_hit = true;
        }
        else if ((offset == 427 && !fire_output_signal)) {
          // output_category_correctness[offset]--;      
          printf("offset 247: %d\n", pred_category);
          output_category_correctness[offset]--;
          output_category_correctness[offset] = std::max(0, output_category_correctness[offset]);
          // is_output_category_matching += output_category_correctness[offset];
          is_output_category_matching--;
          is_output_category_matching = std::max(0, is_output_category_matching.load());
          count_of_wrong_guesses++;  
        }
        else if (offset != 427 && fire_output_signal) {
          is_output_category_matching += 0;
          is_output_category_matching = std::max(0, is_output_category_matching.load());
          count_of_wrong_guesses++;  
        }

        // A constant 1->0->1 or 0->1->1 represent the prediction.
        pred_labels[offset] = int(fire_output_signal);
        cm.num_samples++;
      }
      // if (index % 100000 == 0) {
      //   printf("index: %d \n", index);
      // }
    }

    // After each round of calculation, decide the scale of correct prediction stimulation.
    // if (is_prediction_hit) {
    //   int32_t reaction_signal = init_reaction_signal;
    //   for (int i = 0; i < (IDX_REACTION_RANGE_END - IDX_REACTION_RANGE_START); i++) {
    //     if (i < is_output_category_matching) {
    //       global_array[i + IDX_REACTION_RANGE_START] = pack_word(i + IDX_REACTION_RANGE_START, 0, 0, 1, reaction_signal);
    //     }
    //   }
    //   is_prediction_hit = false;  // Reset is_prediction_hit after each round.
    //   init_reaction_signal = 1 - init_reaction_signal;
    // }

    init_reaction_signal = 1 - init_reaction_signal;
    int32_t next_round_input_range = std::max(10, is_output_category_matching.load());
    load_image_to_manifold(next_round_input_range);  // Control the input range after evaluation. Always have 10 pixels as base inputs.
    printf("predictions made: %d\n", predictions_made.load());
    printf("next round input range: %d\n", next_round_input_range);
    predictions_made = 0;  // Reset the predictions_made.
    is_output_category_matching = 0;  // Reset the output evaluation.

    // After one full round calculation of all nodes in the manifold, calculate the F1-score on error rates and reset the confusion matrix.
    // Run the record of all category outputs in one run.
    record(cm, true_labels, pred_labels);
    // printf("Recorded {pred_labels.count(1)} predictions with 1.");
    summary(cm);
    pred_labels.assign(OUTPUT_SIZE, 0);
    cm = create_confusion_matrix(OUTPUT_SIZE);
    // shuffle();  // Shuffle the order of nodes to execute in the next run!
  // TODO: Migrate the function into GPU and see if magic happens!
  }

  // Always save the gloab_array.
  save_manifold(global_array, MANIFOLD_PATH, DEBUG_PATH);
}