#ifndef _LOADING_IMAGE_H_
#define _LOADING_IMAGE_H_

#include <string>
#include <vector>

#include "constants.h"


bool load_jpeg_to_input_buffer(const std::string& filename, std::vector<int32_t>& out_array);

// TODO: Create a new method loading jpeg in RGB format.
// Using Temporal Pulse to reflect different intensities of photons.
// 3 channels: R, G, B
// First step is using 8-bits to represent 256 range.
// R intensity: 0-31 => 00000000
//              32-63 => 10000000
//              ...
//              224-255 => 11111111
// Hypercube takes 8-clocks (there will be NO refractory period for those special retina nodes) to finish absorbing one image.
// In this process, both 0 and 1 represent excited signals; so 0 represents vertex with -1 status, 1 represents vertex with 1 status.
// The refractory period (1-clock) will still happen after the vertex becomes excited; so eventually it takes 16-clocks to finish absorbing one image.
bool load_jpeg_to_input_buffer_in_rgb_format(const std::string& filename, int32_t (&out_array)[8][RGB_INPUT_BUFFER_SIZE]);
// bool load_jpeg_to_input_buffer_in_rgb_format(const std::string& filename, int32_t (&out_array)[8][3][RGB_INPUT_BUFFER_SIZE]);

#endif
