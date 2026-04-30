#include <iostream>
#include <math.h>
#include <cstdint>  // Required for int32_t
#include <vector>
#include <algorithm> // For std::shuffle
#include <random>    // For std::mt19937 and std::random_device
#include <numeric>   // For std::iota
#include <string>
#include <fstream>

/*
Format:
--------------------------------------
| addr | counter |   k   | dir| state|
| 0-21 |  22-24  | 25-29 | 30 |  31  |
--------------------------------------

For each value in integer format, there are 16 different changes it could be.
*/

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
int32_t IDX_REACTION_RANGE_START = 100000;
int32_t IDX_REACTION_RANGE_END = 101000;

// =========== End of Constants =========

// ========== Global array ======
std::vector<int32_t> global_array(1 << ADDR_BITS);
std::vector<int32_t> cycle_delays_array(1 << ADDR_BITS);
std::vector<int32_t> index_array(1 << ADDR_BITS);

// ========== Manifold file ========
std::string MANIFOLD_PATH = "manifold.txt";
std::string DEBUG_PATH = "debug.txt";

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

// The range where inputs are decided by outputs of manifold.
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

  // Forbid self-recycling connection pair of nodes.
  int32_t target_of_neighbor = unpack_addr(neighbor);

  // Make cycle delays based on connection strength.
  // if (cycle_delays_array[address] > 0) {
  //   cycle_delays_array[address]--;
  //   return w;
  // }

  // Add a new non-linearity into manifold.
  // A very simple logic: 
  //  1. if the state is 1, flip it back to 0 in this run and doing nothing else.
  //  2. if the state is 0, continue the logic.
  if (self_state == 1) {
    // A hacking way, since the state is at the last bit, flipping it from 1 to 0 meaning subtract 1 directly.
    // return w - 1;
    return pack_word(address, strength, k, direction, /*state=*/0);
  }

  // Skip heartbeat over reaction nodes since they will be taken care of separately.
  // TODO: Update cycle delay accordingly, even for output nodes.
  if (is_reaction_range(address)) {
    return w;
  }

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
  cycle_delays_array[address] = 7 - strength;  // Update the cycle_delay for the word based on its connection strength.

  // Change to next index if the counter strength is 0.
  // TODO: Remove the prohibition of self-pulling synchronization between two nodoes where two connections only connect with each other.
  //       This will be replaced by random ordering in single thread CPU mode.
  //       && (address == target_of_neighbor)
  if (strength == 0) {
    strength = 0;
    // printf("k, direction before change: %d, %d\n", k, direction);
    decide_k_and_dirction(k, direction);
    // printf("k, direction after change: %d, %d\n", k, direction);
  }

  return pack_word(address, strength, k, direction, self_state);
}
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
          global_array[idx_dimension0] = pack_word(idx_dimension0, 0, 0, 1, (r >> bit_pos) & 1);
          idx_dimension0++;
        } else {
          global_array[idx_dimension1] = pack_word(idx_dimension1, 0, 0, 1, (r >> bit_pos) & 1);
          idx_dimension1++;
        }
      }

      for (int bit_pos = 7; bit_pos >= 0; bit_pos--) {
        if (is_dimension_0) {
          global_array[idx_dimension0] = pack_word(idx_dimension0, 0, 0, 1, (g >> bit_pos) & 1);
          idx_dimension0++;
        } else {
          global_array[idx_dimension1] = pack_word(idx_dimension1, 0, 0, 1, (g >> bit_pos) & 1);
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
  // Calculate the count of nodes with state 1.
  int32_t active_state_count = 0;
  for (int32_t word : global_array) {
    if (word & 1) {
      active_state_count++;
    }
  }
  printf("*** Active node count: {%d} ***\n", active_state_count);

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
  // Calculate the count of nodes with state 1.
  int32_t active_state_count = 0;
  for (int32_t word : global_array) {
    if (word & 1) {
      active_state_count++;
    }
  }
  printf("=======\n PreRun Active node count: {%d} \n=======\n", active_state_count);
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
  for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--use_save_data") {
        use_save_data = true;
      }
      if (arg == "--runs" && i + 1 < argc) {
        maximum_runs = static_cast<int32_t>(std::atoi(argv[++i]));
      }
  }

  // Init the global_array.
  bool need_init = false;
  if (use_save_data) {
    need_init = !load_manifold(global_array, MANIFOLD_PATH);
  }
  printf("Do we need to re-init manifold array: %d\n", need_init);

  for (int i = 0; i < (1 << ADDR_BITS); i++) {
    if (need_init) {
      global_array[i] = pack_word(i, 0, 0, 1, 0);
    }
    cycle_delays_array[i] = 7;  // Start with maximum cycle delay when the connection strength is 0.
    index_array[i] = i;
  }
  load_image_to_manifold(INPUT_SIZE);

  int32_t n = 1 << ADDR_BITS;

  ConfusionMatrix cm = create_confusion_matrix(OUTPUT_SIZE);
  std::vector<int32_t> true_labels(OUTPUT_SIZE);
  std::vector<int32_t> pred_labels(OUTPUT_SIZE);

  int32_t is_output_category_matching = 0;
  int32_t predictions_made = 0;
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

  shuffle();  // Do a shuffle before the execution!
  int count = 0;
  pre_run_summary();
  while (count++ < maximum_runs) {
    for (int index = 0; index < n; index++) {
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
        bool fire_output_signal = counter == 2;  // The connection strength represents the output signal.

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
          is_output_category_matching += 1;
          is_prediction_hit = true;
        }
        else if ((offset == 427 && !fire_output_signal)) {
          // output_category_correctness[offset]--;      
          printf("offset 247: %d\n", pred_category);
          output_category_correctness[offset]--;
          output_category_correctness[offset] = std::max(0, output_category_correctness[offset]);
          // is_output_category_matching += output_category_correctness[offset];
          is_output_category_matching--;
          count_of_wrong_guesses++;  
        }
        else if (offset != 427 && fire_output_signal) {
          is_output_category_matching--;
          is_output_category_matching = std::max(0, is_output_category_matching);
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
    // load_image_to_manifold(is_output_category_matching);  // Control the input range after evaluation.
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
    shuffle();  // Shuffle the order of nodes to execute in the next run!
  // TODO: Migrate the function into GPU and see if magic happens!
  }

  if (use_save_data) {
    save_manifold(global_array, MANIFOLD_PATH, DEBUG_PATH);
  }
}