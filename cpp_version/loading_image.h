#ifndef _LOADING_IMAGE_H_
#define _LOADING_IMAGE_H_

#include <string>
#include <vector>


bool load_jpeg_to_input_buffer(const std::string& filename, std::vector<int32_t>& out_array);

#endif
