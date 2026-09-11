%%writefile debugging.h

#ifndef _DEBUGGING_H_
#define _DEBUGGING_H_


#include <cstdio>
#include <fstream>

void save_to_disk(const char* filename, int* h_data, size_t count);

void save_to_disk(const char* filename, uint8_t* h_data, size_t count);

void print_dimension_distribution(
  const bool* h_excited,
  const int *energy,
  const bool *vertex_internal_state,
  int hypercube_dim = 22);

#endif
