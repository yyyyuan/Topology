#include <cstdint>  // Required for int32_t

#include "constants.h"
#include "database.h"
#include "manifold_operators.h"


// This function currently only loads one image into manifold.
// Hence signals in input nodes don't need to change.
// TODO: Upgrade this function so it can read tons of data from sources such as ImageNet.
// TODO: Change the input formats from constant 1s into 0<->1 pulse to mimic input signals.
void load_image_to_manifold(int32_t input_range);

bool is_input_range(int32_t index);