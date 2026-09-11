%%writefile debugging.cu

#include <fstream>
#include <iomanip>  // Required for std::setw
#include <iostream>
#include <vector>

#include "debugging.h"

void save_to_disk(const char* filename, int* h_data, size_t count) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        //file.write(reinterpret_cast<const char*>(h_data), count * sizeof(int));
        //file.close();
        for (size_t i = 0; i < count; ++i) {
          file << h_data[i] << "\n"; // Formats integers into human-readable ASCII numbers
        }
    }
}

void save_to_disk(const char* filename, uint8_t* h_data, size_t count) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Correct multiplier: sizeof(uint8_t) is 1 byte
        // file.write(reinterpret_cast<const char*>(h_data), count * sizeof(uint8_t));
        for (size_t i = 0; i < count; ++i) {
            // Cast to int forces stream to format as numeric text (e.g., "255\n")
            file << static_cast<int>(h_data[i]) << "\n";
        }
    }
}

void print_dimension_distribution(
  const bool* h_excited,
  const int *energy,
  const bool *vertex_internal_state,
  int hypercube_dim) {
    std::vector<int> excited_counts(hypercube_dim + 1, 0);
    std::vector<int> positive_excited_counts(hypercube_dim + 1, 0);
    std::vector<int> positive_min_energy(hypercube_dim + 1, 0);
    std::vector<int> positive_max_energy(hypercube_dim + 1, 0);
    std::vector<int> positive_energy_counts(hypercube_dim + 1, 0);
    std::vector<int> negative_excited_counts(hypercube_dim + 1, 0);
    std::vector<int> negative_energy_counts(hypercube_dim + 1, 0);
    std::vector<int> negative_min_energy(hypercube_dim + 1, 0);
    std::vector<int> negative_max_energy(hypercube_dim + 1, 0);

    for (uint32_t idx = 0; idx < (1U << hypercube_dim); ++idx) {
      int dist = __builtin_popcount(idx); // Counts 1-bits in address
      if (h_excited[idx]) {
        excited_counts[dist]++;
      }

      if (positive_min_energy[dist] == 0) {
        if (vertex_internal_state[idx]) {
          positive_min_energy[dist] = energy[idx];
        }
        else {
          negative_min_energy[dist] = energy[idx];
        }
      }

      if (energy[idx] > 1) {
        if (vertex_internal_state[idx]) {
          positive_energy_counts[dist]++;
          positive_min_energy[dist] = std::min(positive_min_energy[dist], energy[idx]);
          positive_max_energy[dist] = std::max(positive_max_energy[dist], energy[idx]);

          if (h_excited[idx]) {
            positive_excited_counts[dist]++;
          }
        }
        else {
          negative_energy_counts[dist]++;
          negative_min_energy[dist] = std::min(negative_min_energy[dist], energy[idx]);
          negative_max_energy[dist] = std::max(negative_max_energy[dist], energy[idx]);

          if (h_excited[idx]) {
            negative_excited_counts[dist]++;
          }
        }
      }
    }

    static int col_width = 16;
    std::cout << "\n--- Excited Vertices by Hypercube Layer (Hamming Distance) ---\n";
    std::cout << "| " << std::setw(col_width) << std::left << "Layer"
              << " | " << std::setw(col_width) << std::left << "Excited counts all"
              << " | " << std::setw(col_width) << std::left << "Excited counts pos"
              << " | " << std::setw(col_width) << std::left << "Count (>= 2) Pos"
              << " | " << std::setw(col_width) << std::left << "Min Energy Pos"
              << " | " << std::setw(col_width) << std::left << "Max Energy Pos"
              << " | " << std::setw(col_width) << std::left << "Excited counts neg"
              << " | " << std::setw(col_width) << std::left << "Count (>= 2) Neg"
              << " | " << std::setw(col_width) << std::left << "Min Energy Neg"
              << " | " << std::setw(col_width) << std::left << "Max Energy Neg"
              << " |\n";
    for (int k = 0; k <= hypercube_dim; ++k) {
        std::cout << "| " << std::setw(col_width) << std::right << k
                  << " | " << std::setw(col_width) << std::right << excited_counts[k]
                  << " | " << std::setw(col_width) << std::right << positive_excited_counts[k]
                  << " | " << std::setw(col_width) << std::right << positive_energy_counts[k]
                  << " | " << std::setw(col_width) << std::right << positive_min_energy[k]
                  << " | " << std::setw(col_width) << std::right << positive_max_energy[k]
                  << " | " << std::setw(col_width) << std::right << negative_excited_counts[k]
                  << " | " << std::setw(col_width) << std::right << negative_energy_counts[k]
                  << " | " << std::setw(col_width) << std::right << negative_min_energy[k]
                  << " | " << std::setw(col_width) << std::right << negative_max_energy[k]
                  << " |\n";
    }
}

