#ifndef _LOADING_IMAGE_H_
#define _LOADING_IMAGE_H_

#include <string>
#include <vector>


bool loadJpegTo64x64Array(const std::string& filename, std::vector<int32_t>& out_array);

#endif
