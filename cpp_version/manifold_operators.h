// This file defines all operatoros related to words inside the manifold.

#include <cstdint>  // Required for int32_t

// ========== Operators related to words =========
int32_t pack_word(int32_t addr, int32_t counter, int32_t k, int32_t dir_bit, int32_t state);

int32_t unpack_addr(int32_t word);

int32_t unpack_counter(int32_t word);

int32_t unpack_k(int32_t word);

int32_t unpack_dir(int32_t word);

int32_t unpack_state(int32_t word);

int32_t with_addr(int32_t word, int32_t logical_addr);

int32_t with_strength(int32_t word, int32_t strength);

int32_t with_k(int32_t word, int32_t k);

int32_t with_direction(int32_t word, int32_t direction);

int32_t with_state(int32_t word, int32_t state);

// 1 if 1-bit states differ, else 0.
int32_t resonates(int32_t self_state, int32_t neighbor_state);

void decide_k_and_dirction(int32_t& k, int32_t& direction);

int32_t update_k_and_direction_within_word(int32_t word);

bool is_input_range(int32_t index);

bool is_output_range(int32_t index);

// The range where inputs are decided by outputs of manifold.
bool is_reaction_range(int32_t index);

// ============================== //
// The heartbeat function is the core of algorithm.
int32_t heartbeat(int32_t w);