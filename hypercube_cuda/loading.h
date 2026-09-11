%%writefile loading.h

#ifndef _LOADING_H_
#define _LOADING_H_

int load_image(std::string image_path, uint8_t*& d_raw_rgb, size_t& total_elements);

#endif