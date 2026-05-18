#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <cstdint>  // Required for int32_t
#include <string>

// Constants used in manifold.
inline constexpr int32_t ADDR_BITS = 22;
inline constexpr int32_t K_DIM = 22;  // hypercube dimension for logical addresses

inline constexpr int32_t SHIFT_ADDR = 1 + 1 + 5 + 3;
inline constexpr int32_t SHIFT_COUNTER = 1 + 1 + 5;
inline constexpr int32_t SHIFT_K = 1 + 1;
inline constexpr int32_t SHIFT_DIR = 1;
inline constexpr int32_t SHIFT_STATE = 0;

inline constexpr int32_t MASK_ADDR = (1 << (ADDR_BITS)) - 1;
inline constexpr int32_t MASK_COUNTER = 0b111;
inline constexpr int32_t MASK_K = 0b11111;
inline constexpr int32_t MASK_DIR = 0b1;
inline constexpr int32_t MASK_STATE = 0b1;

inline constexpr int32_t U32_MASK = 0xFFFFFFFF;
// Only bits related to k are set to 0, 0b11111111111111111111111110000011 <- 32-bits
inline constexpr int32_t K_MASK_IN_WORD = 0xFFFFFF83;
// Only bits related to directions are set to 0, 0b11111111111111111111111111111101 <- 32-bits
inline constexpr int32_t DIRECTION_MASK_IN_WORD = 0xFFFFFFFD;

// Bit 30: 1 = increase k when strength hits 0; 0 = decrease k (with wrap in 5 bits).
inline constexpr int32_t DIR_INCREASE_K = 1;
inline constexpr int32_t DIR_DECREASE_K = 0;

// ========= Parameters related to I/O (image read) =========
inline constexpr int32_t IMAGE_WIDTH = 64;
inline constexpr int32_t IMAGE_HEIGHT = 64;
inline constexpr int32_t CHANNELS = 3; // RGB has 3 channcels in each pixel.
inline constexpr int32_t BITS_PER_CHANNEL = 8;

inline constexpr int32_t INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS * BITS_PER_CHANNEL;  // 98304 = 64 * 64 * 3 * 8 bits for one image.
inline constexpr int32_t OUTPUT_SIZE = 1000; // Let's do 1000 different categories.

inline const std::string IMAGE_PATH = "test_image.png";

// The range of Reaction must be larger than 2024 and out of input dimension range.
// The maximum range means there is very long history of correct predictions.
inline constexpr int32_t IDX_REACTION_RANGE_START = 100000;
inline constexpr int32_t IDX_REACTION_RANGE_END = 101000;

// ========== Manifold file ========
inline const std::string MANIFOLD_PATH = "manifold.txt";
inline const std::string DEBUG_PATH = "debug.txt";

#endif
