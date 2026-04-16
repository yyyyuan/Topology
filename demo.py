from __future__ import annotations

"""Hypercube 32-bit field cells: packed word layout, neighbor addressing, and step logic.

Word layout (uint32, little-endian bit numbering as usual):

- Bits 0–21: logical address (canonical source is array index; low bits of the word should match).
- Bits 22–24: 3-bit counter / connection strength (0–7), saturating.
- Bits 25–29: neighbor index k (5 bits); geometry uses k_eff = k % 22 on a 22-bit cube.
- Bit 30: search direction — 0 = decrease k (with wrap in 0..21), 1 = increase k.
- Bit 31: field state (used for resonance vs neighbor state only).

One *macro* step reads from ``prev`` only and writes ``next``. Each cell runs an inner loop:
pull neighbor state bit → if states differ (resonate) counter += 1 else counter -= 1 (saturated) →
if counter == 0, advance k per bit 30 and stop the inner loop; else repeat. Inner iterations are
capped by ``max_micro_steps``.
"""

"""
Format:
--------------------------------------
| addr | counter |   k   | dir| state|
| 0-21 |  22-24  | 25-29 | 30 |  31  |
--------------------------------------

For each value in integer format, there are 16 different changes it could be.

"""
from PIL import Image

import numpy as np
from error_rate import (
    create_confusion_matrix,
    record,
    f1_score,
    summary
)

ADDR_BITS = 22
K_DIM = 22  # hypercube dimension for logical addresses

SHIFT_ADDR = 1 + 1 + 5 + 3
SHIFT_COUNTER = 1 + 1 + 5
SHIFT_K = 1 + 1
SHIFT_DIR = 1
SHIFT_STATE = 0

MASK_ADDR = (1 << (ADDR_BITS + 1)) - 1
MASK_COUNTER = 0b111
MASK_K = 0b11111
MASK_DIR = 0b1
MASK_STATE = 0b1

U32_MASK = 0xFFFFFFFF
# Only bits related to k are set to 0, 0b11111111111111111111111110000011 <- 32-bits
K_MASK_IN_WORD = 0xFFFFFF83
# Only bits related to directions are set to 0, 0b11111111111111111111111111111101 <- 32-bits
DIRECTION_MASK_IN_WORD = 0xFFFFFFFD

# Bit 30: 1 = increase k when strength hits 0; 0 = decrease k (with wrap in 5 bits).
DIR_INCREASE_K = 1
DIR_DECREASE_K = 0

# ========= Parameters related to I/O (image read) =========
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3 # RGB has 3 channcels in each pixel.
BITS_PER_CHANNEL = 8

INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS * BITS_PER_CHANNEL  # 98304 = 64 * 64 * 3 * 8 bits for one image.
OUTPUT_SIZE = 1000 # Let's do 1000 different categories.

IMAGE_PATH = "test_image.png"

IDX_DIMENSION_0_RANGE = IMAGE_HEIGHT * IMAGE_WIDTH // 2
IDX_DIMENSION_1_RANGE = 2**(ADDR_BITS - 1) + IDX_DIMENSION_0_RANGE # 2^(ADDR_BITS - 1) + Range
# ========= End of I/O parameters ========

global_array = []


def u32(x: int) -> int:
    """Unsigned 32-bit: keep low 32 bits."""
    return x & U32_MASK


def pack_word(addr: int, counter: int, k: int, dir_bit: int, state: int) -> int:
    """Pack fields into one uint32 word (masked)."""
    return u32(
        (state & 1)
        | ((dir_bit & MASK_DIR) << SHIFT_DIR)
        | ((k & MASK_K) << SHIFT_K)
        | ((counter & MASK_COUNTER) << SHIFT_COUNTER)
        | ((addr & MASK_ADDR) << SHIFT_ADDR)
    )

def unpack_addr(word: int) -> int:
    return (u32(word) >> SHIFT_ADDR) & MASK_ADDR


def unpack_counter(word: int) -> int:
    return (u32(word) >> SHIFT_COUNTER) & MASK_COUNTER


def unpack_k(word: int) -> int:
    return (u32(word) >> SHIFT_K) & MASK_K


def unpack_dir(word: int) -> int:
    return (u32(word) >> SHIFT_DIR) & 1


def unpack_state(word: int) -> int:
    return u32(word) & MASK_STATE

def with_addr(word: int, logical_addr: int) -> int:

    # return (word & ((1 << SHIFT_ADDR) - 1)) | ((logical_addr & MASK_ADDR) << SHIFT_ADDR)
    return (word & ~(MASK_ADDR << SHIFT_ADDR)) | ((logical_addr & MASK_ADDR) << SHIFT_ADDR)


def with_strength(word: int, strength: int) -> int:
    return (word & ~(MASK_COUNTER << SHIFT_COUNTER)) | (
        (strength & MASK_COUNTER) << SHIFT_COUNTER
    )


def with_k(word: int, k: int) -> int:
    return (word & ~(MASK_K << SHIFT_K)) | ((k & MASK_K) << SHIFT_K)


def with_direction(word: int, direction: int) -> int:
    return (word & ~(MASK_DIR << SHIFT_DIR)) | (
        (direction & MASK_DIR) << SHIFT_DIR
    )


def with_state(word: int, state: int) -> int:
    return (word & ~(MASK_STATE << SHIFT_STATE)) | (
        (state & MASK_STATE) << SHIFT_STATE
    )


def resonates(self_state: int, neighbor_state: int) -> int:
    """1 if 1-bit states differ, else 0."""
    return (self_state ^ neighbor_state) & 1

def decide_k_and_dirction(k: int, direction: int):
    if k == 0 and direction != DIR_INCREASE_K:
        direction = DIR_INCREASE_K
    if k == ADDR_BITS and direction == DIR_INCREASE_K:
        direction = DIR_DECREASE_K

    if direction == DIR_INCREASE_K:
        k = (k + 1) & MASK_K
    else:
        k = (k - 1) & MASK_K

    return k, direction

def update_k_and_direction_within_word(word: int) -> int:
    w = u32(word)
    direction = unpack_dir(w)
    k = unpack_k(w)

    k, direction = decide_k_and_dirction(k, direction)

    return w & K_MASK_IN_WORD & DIRECTION_MASK_IN_WORD | ((k & MASK_K) << SHIFT_K) | ((direction & MASK_DIR) << SHIFT_DIR)

def one_micro(word: int) -> int:
    """Single pull: update counter from resonance; if counter is 0, advance k. State unchanged."""
    w = u32(word)
    address = unpack_addr(word)
    direction = unpack_dir(w)
    k = unpack_k(w)
    strength = unpack_counter(w)
    self_state = unpack_state(w)

    k = min(k, ADDR_BITS) # In current implementation, the maximum value of k is 22

    neighbor_index = address ^ (1 << k) # Flip the bit at k
    neighbor = global_array[neighbor_index]

    # Output node cannot be used as source to pull data from.
    # Instead we update sections k and direction inside the word directly.
    if (is_output_range(neighbor_index)):
        return update_k_and_direction_within_word(w)

    neighbor_state = unpack_state(neighbor)
    resonate = self_state != neighbor_state

    if resonate:
        strength = min(7, strength + 1)
        self_state = neighbor_state
    else:
        strength = max(0, strength - 1)

    # Change to next index if the counter strength is 0.
    if strength == 0:
        k, direction = decide_k_and_dirction(k, direction)

    out = pack_word(address, strength, k, direction, self_state)

    return out

def print_word(word):
    print("=========")
    print(f"{word}")
    print("logical_address: ", unpack_addr(word))
    print("counter: ", unpack_counter(word))
    print("K: ", unpack_k(word))
    print("direction: ", unpack_dir(word))
    print("state: ", unpack_state(word))
    print("neighbor address: ", unpack_addr(word) + (1 << unpack_k(word)))

def print_float(word):
    # Create an array with a 32-bit integer
    int_arr = np.array([word], dtype=np.uint32)

    # Reinterpret the bits as a 32-bit float
    float_arr = int_arr.view(np.float32)

    print(float_arr[0])  # Output: 5.199999809265137 (approx 5.2)

def load_image_to_manifold(image_path):
    # This function currently only loads one image into manifold.
    # Hence signals in input nodes don't need to change.
    # TODO: Upgrade this function so it can read tons of data from sources such as ImageNet.
    image = Image.open(image_path).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    idx_dimension0 = 0
    idx_dimension1 = len(global_array) // 2 # Idx must be integer
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):                
            r, g, b = image.getpixel((x, y))
            is_dimension_0 = x < IMAGE_WIDTH // 2
            for channel_val in (r, g, b):
                for bit_pos in range(7, -1, -1):
                    if is_dimension_0:
                        global_array[idx_dimension0] = pack_word(idx_dimension0, 1, 0, 1, (channel_val >> bit_pos) & 1)
                        idx_dimension0 += 1
                    else:
                        global_array[idx_dimension1] = pack_word(idx_dimension1, 1, 0, 1, (channel_val >> bit_pos) & 1)
                        idx_dimension1 += 1

def load_image_to_manifold_with_idx(index):
    # TODO: When this function is called, the `image` should already be loaded somewhere else rather than in this function.
    image = Image.open("not_found_path").convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    is_idx_dimension_0_range = index < IDX_DIMENSION_0_RANGE

    if is_idx_dimension_0_range:
        y = index // (IMAGE_WIDTH // 2)
        x = index % (IMAGE_WIDTH // 2)
    else:
        y = (index - 2**(ADDR_BITS - 1)) // (IMAGE_WIDTH // 2)
        x = (index - 2**(ADDR_BITS - 1)) % (IMAGE_WIDTH // 2) + 2**(ADDR_BITS - 1)
    r, g, b = image.getpixel((x, y))
    for channel_val in (r, g, b):
        for bit_pos in range(7, -1, -1):
            if is_idx_dimension_0_range:
                global_array[index] = pack_word(index, 1, 0, 1, (channel_val >> bit_pos) & 1)
            else:
                global_array[index] = pack_word(index, 1, 0, 1, (channel_val >> bit_pos) & 1)

def is_input_range(index):
    if (index >= 0 and index < IDX_DIMENSION_0_RANGE) or (index >= 2**(ADDR_BITS - 1) and index < IDX_DIMENSION_1_RANGE):
        return True
    return False

def is_output_range(index):
    if (index >= (2**ADDR_BITS - OUTPUT_SIZE) and index < 2**ADDR_BITS):
        return True
    return False

if __name__ == "__main__":
    # Init the global_array.
    for i in range(2**ADDR_BITS):
        global_array.append(pack_word(i, 1, 0, 1, 0))

    # TODO: Init the manifold to accept RGB images and output its categorizations.
    # TODO: Mark nodes in the region of input images as ones not listening to other nodes but only image inputs.
    load_image_to_manifold(IMAGE_PATH)

    index = 0
    n = len(global_array)
    print("length: ", len(global_array))

    # Confusion matrix and F1 score is used in output precision calculation.
    cm = create_confusion_matrix(OUTPUT_SIZE)
    true_labels = [0] * OUTPUT_SIZE
    pred_labels = [0] * OUTPUT_SIZE

    # For one single image, there is only 1 matching category out of all 1000 categories.
    # TODO: Eventually this should be modified by datasets, instead of being hardcoded.
    true_labels[427] = 1

    # TODO: The manifold is able to accept arbitrary signals other than standard image inputs.
    #       Those arbitrary signals are vital to evolve the manifold to generate expected outputs.
    #       It's easy to see such architecture is extendable, to meet all different types of real-world (physical) requirements.

    while True:
        if (is_input_range(index)):
            # Increment over input nodes, which don't pull signals from other nodes.
            index = (index + 1) % n
            continue

        # The core calculation.
        updated_word = one_micro(global_array[index])
        global_array[index] = updated_word
        
        # print(f"Index {index} updated to {global_array[index]}")

        if (is_output_range(index)):
            pred_category = unpack_state(u32(updated_word))

            # The offset of prediction array is (2**ADDR_BITS - OUTPUT_SIZE)
            pred_labels[index - (2**ADDR_BITS - OUTPUT_SIZE)] = pred_category
        
        # Increment and wrap around
        index = (index + 1) % n

        if (index % 100000 == 0):
            print("index: ", index)

        # After one full round calculation of all nodes in the manifold, calculate the F1-score on error rates and reset the confusion matrix.
        if index == (n - 1):
            # Run the record of all category outputs in one run.
            record(cm, true_labels, pred_labels)
            print(f"Recorded {pred_labels.count(1)} predictions with 1.")
            summary(cm)
            predictions = []
            cm = create_confusion_matrix(OUTPUT_SIZE)

        # TODO: Migrate the function into GPU and see if magic happens!




    # TODO: Test the new manifold with image recognizations, similar to AlexNet.
    # 1. Add input porotals accepts binary signals from images
    # 2. Add output portals generating categorizations
    # 3. Add a new input portal accepting stimulation signals when the output categorization is correct.
    #    This is supposed to affect the manifold evolution.
    # 4. Add new cross-entropy feature validating the output/categorization accuracies of this manifold.
    #    This can follow the standard accuracy calculation in back-propagation methods,
    #    the only difference here is that here we used an active manifold to do evaluatoin.