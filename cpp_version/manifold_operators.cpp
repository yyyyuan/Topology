#include <cstdint>  // Required for int32_t

#include "constants.h"
#include "database.h"
#include "manifold_operators.h"

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

    return (word & K_MASK_IN_WORD & DIRECTION_MASK_IN_WORD) | ((k & MASK_K) << SHIFT_K) | ((direction & MASK_DIR) << SHIFT_DIR);
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
    strength = 1;  // Base counter is set to 1 instead of 0 to achieve the "active vacuum" idea.
    self_state = 1;
    // printf("k, direction before change: %d, %d\n", k, direction);
    decide_k_and_dirction(k, direction);
    // printf("k, direction after change: %d, %d\n", k, direction);
  }

  return pack_word(address, strength, k, direction, self_state);
}