// A new program running the manifold using the new vertex design.
// Putting everything in a new file, meaning a new start I guess.

#include <cstdint> // Required for int32_t
#include <filesystem> // Required for std::filesystem
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

#include "classifier.h"
#include "constants.h"
#include "database.h"
#include "loading_image.h"
#include "vertex.h"

std::unordered_map<int32_t, Pattern> classifier = {};
int32_t best_matched_category = -1;
float highest_probability_score = 0;
std::vector<float> probability_score_list(CATEGORY_COUNT);

std::vector<std::string> return_all_file_names(std::string directory_path) {
    std::vector<std::string> file_names = {};
    try {
        if (std::filesystem::exists(directory_path) && std::filesystem::is_directory(directory_path)) {
            // Loop through all items in the directory
            for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
                // Check if it's a regular file (skips sub-folders)
                if (entry.is_regular_file()) {
                    // 1. Get the extension of the file
                    std::filesystem::path ext = entry.path().extension();
                    
                    // 2. Only insert if the extension matches ".jpeg" or ".jpg"
                    if (ext == ".JPEG" || ext == ".jpeg" || ext == ".jpg") {
                        file_names.push_back(entry.path().filename().string());
                        std::cout << entry.path().filename() << "\n";
                    }
                }
            }
            
        } else {
            std::cerr << "Directory does not exist.\n";
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return file_names;
}

// TODO: Now the hypercube should systematically import images into the hypercube.
void build_input_array() {
    // for (int y = 0; y < IMAGE_HEIGHT; y++) {
    //     for (int x = 0; x < IMAGE_WIDTH; x++) {
    //         bool t = (x + y) > IMAGE_HEIGHT;
    //         input_array[y * IMAGE_WIDTH + x] = t;
    //     }
    // }
    // std::string image_path = "images/photo.jpg";
    // if (load_jpeg_to_input_buffer(image_path, input_array)) {
    //     std::cout << "Successfully decoded and downsampled JPEG into 64x64 grid!" << std::endl;
    //     std::cout << "Top-left pixel hex (ARGB): 0x" << std::hex << input_array[0] << std::endl;
    // }

    for (int i = 0; i < CATEGORY_COUNT; i++) {
        std::string dir_path = "images/training/" + std::to_string(i);
        std::vector<std::string> file_names = return_all_file_names(dir_path);
        for (const std::string& file_name : file_names) {
            std::string image_path = dir_path + "/" + file_name;
            std::cout << image_path << std::endl;
            if (load_jpeg_to_input_buffer(image_path, input_buffer[i])) {
                std::cout << "Successfully decoded and downsampled JPEG into 64x64 grid!" << std::endl;
                std::cout << "Top-left pixel hex (ARGB): 0x" << std::hex << input_buffer[i][0] << std::endl;
            }
        }
    }

    // image_path = "images/photo.jpg";
    // if (load_jpeg_to_input_buffer(image_path, input_buffer[0])) {
    //     std::cout << "Successfully decoded and downsampled JPEG into 64x64 grid!" << std::endl;
    //     std::cout << "Top-left pixel hex (ARGB): 0x" << std::hex << input_buffer[0][0] << std::endl;
    // }

    // image_path = "images/image2.jpg";
    // if (load_jpeg_to_input_buffer(image_path, input_buffer[1])) {
    //     std::cout << "Successfully decoded and downsampled JPEG into 64x64 grid!" << std::endl;
    //     std::cout << "Top-left pixel hex (ARGB): 0x" << std::hex << input_buffer[1][0] << std::endl;
    // }

    input_array_ptr = &input_buffer[0];
}

void import_validation_image() {
    std::string dir_path = "images/val";
    std::vector<std::string> file_names = return_all_file_names(dir_path);
    for (const std::string& file_name : file_names) {
        std::string image_path = dir_path + "/" + file_name;
        std::cout << image_path << std::endl;
        if (load_jpeg_to_input_buffer(image_path, validation_img_buffer[0])) {
            std::cout << "Successfully decoded and downsampled JPEG into 64x64 grid!" << std::endl;
            std::cout << "Top-left pixel hex (ARGB): 0x" << std::hex << input_buffer[0][0] << std::endl;
        }
    }
}

struct HammerStringDebugUnit {
    int32_t min_energy_positive;
    int32_t max_energy_positive;
    int32_t count_positive;
    int32_t excited_vertexes_count_positive;

    int32_t min_energy_negative;
    int32_t max_energy_negative;
    int32_t count_negative;
    int32_t excited_vertexes_count_negative;

};

// Calculate the count of nodes with state 1 in the hypercube.
int32_t analyze_hypercube() {
    std::map<int32_t, int32_t> fibonacci_bucket;
    std::vector<HammerStringDebugUnit> hypercube_structure(ADDR_BITS+1);  // Each layer contains minimal_energy, maximum_energy, number of nodes with energy >= 2.
    int32_t active_state_count = 0;
    int32_t excited_vertexes_count = 0;
    std::map<int32_t, int32_t> excited_vertexes_fibonacci_bucket;
    for (Vertex vertex : hypercube_array) {
        if (vertex.internal_state) {
            active_state_count++;
        }
        fibonacci_bucket[vertex.upper_excite_thresold]++;

        // bit counting
        int active_bits = __builtin_popcount(vertex.address);
        if (hypercube_structure[active_bits].min_energy_positive == 0) {
            if (vertex.internal_state) {
                hypercube_structure[active_bits].min_energy_positive = vertex.energy;
            }
            else {
                hypercube_structure[active_bits].min_energy_negative = vertex.energy;
            }
        }

        if (vertex.energy > 1 && vertex.type != VertexType::INPUT) {
            if (vertex.internal_state) {
                hypercube_structure[active_bits].min_energy_positive = std::min(hypercube_structure[active_bits].min_energy_positive, vertex.energy);
                hypercube_structure[active_bits].max_energy_positive = std::max(hypercube_structure[active_bits].max_energy_positive, vertex.energy);
                hypercube_structure[active_bits].count_positive++;

                if (vertex.excited) {
                    hypercube_structure[active_bits].excited_vertexes_count_positive++;
                }
            }
            else {
                hypercube_structure[active_bits].min_energy_negative = std::min(hypercube_structure[active_bits].min_energy_negative, vertex.energy);
                hypercube_structure[active_bits].max_energy_negative = std::max(hypercube_structure[active_bits].max_energy_negative, vertex.energy);
                hypercube_structure[active_bits].count_negative++;

                if (vertex.excited) {
                    hypercube_structure[active_bits].excited_vertexes_count_negative++;
                }
            }
        }

        if (vertex.excited) {
            excited_vertexes_count++;
            excited_vertexes_fibonacci_bucket[vertex.upper_excite_thresold]++;
        }
    }

    static int col_width = 16;
    std::printf("\nHypercube structure in the format of Hammer String");
    std::printf("\n===========\n");
    std::cout << "| " << std::setw(col_width) << std::left << "Hammer String"
              << " | " << std::setw(col_width) << std::left << "Min Energy Pos"
              << " | " << std::setw(col_width) << std::left << "Max Energy Pos"
              << " | " << std::setw(col_width) << std::left << "Count (>= 2) Pos"
              << " | " << std::setw(col_width) << std::left << "Excited Cnt Pos"
              << " | " << std::setw(col_width) << std::left << "Min Energy Neg"
              << " | " << std::setw(col_width) << std::left << "Max Energy Neg"
              << " | " << std::setw(col_width) << std::left << "Count (>= 2) Neg" 
              << " | " << std::setw(col_width) << std::left << "Excited Cnt Neg"
              << " |\n";
    for (int i = 0; i <= ADDR_BITS; ++i) {
        std::cout << "| " << std::setw(col_width) << std::right << std::dec << i
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i].min_energy_positive
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i].max_energy_positive
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i].count_positive
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i].excited_vertexes_count_positive
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i].min_energy_negative
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i].max_energy_negative
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i].count_negative
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i].excited_vertexes_count_negative
                  << " |\n";
    }
    std::printf("===========\n");

    std::printf("\nEnergy allocation");
    // for (const auto& [bucket, count] : fibonacci_bucket) {
    //     std::printf("%d: %d\n", bucket, count);
    // }

    std::printf("Excited vertexes count: %d\n", excited_vertexes_count);
    for (const auto& [bucket, count] : excited_vertexes_fibonacci_bucket) {
        std::printf("%d: %d\n", bucket, count);
    }

    // TODO: To better understand hypercube properties, we also want to log the addresses of active vertexes.

    return active_state_count;
}

void record() {
    printf("*** Active vertex count: {%d} ***\n", analyze_hypercube());
}

int main(int argc, char *argv[])
{

    bool use_save_data = false;
    int32_t maximum_runs = 10;
    bool need_init = false;
    bool verbose = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--use_save_data")
        {
            use_save_data = true;
        }
        if (arg == "--runs" && i + 1 < argc)
        {
            maximum_runs = static_cast<int32_t>(std::atoi(argv[++i]));
        }
        if (arg == "--init")
        {
            need_init = true;
        }
        if (arg == "--verbose") {
            verbose = true;
        }
    }

    build_input_array();
    import_validation_image();
    output_array[0] = false;

    // Initialize the global_array
    for (int i = 0; i < (1 << ADDR_BITS); i++)
    {
        if (need_init)
        {
            // Allocate INPUT/NORMAL/OUTPUT vertexes.
            VertexType type = VertexType::NORMAL;
            if (i < TARGET_WIDTH * TARGET_HEIGHT) {
                type = VertexType::INPUT;
            }
            // Select 1000 slots in the middle of hypercube as output slots.
            // TODO: It looks like output slots must be very close to input slots to build internal connections because input range is small.
            if (i >= ((1 << (ADDR_BITS - 10)) - OUTPUT_SIZE) && (i < (1 << (ADDR_BITS - 10)))) {
                type = VertexType::OUTPUT;
            }

            hypercube_array[i] = Vertex{
                .address = i,
                .type = type};
        }
    }
    record();
    // printf("output_array status before the run: {%d} \n", output_array[0]);

    int32_t input_source_idx = 0;

    while (true) {
        int32_t count = 0;
        // Reset input source before each run.
        input_array_ptr = &input_buffer[input_source_idx++];
        input_source_idx %= CATEGORY_COUNT;

        best_matched_category = -1;
        highest_probability_score = 0;
        probability_score_list.assign(CATEGORY_COUNT, 0.0f);

        std::printf("The current img category is: %d. ", input_source_idx);
        while (count++ < maximum_runs)
        {
            
            // input_source_idx = count >= maximum_runs ? 1 : 0;
            for (int i = 0; i < hypercube_array.size(); i++)
            {
                // debug(hypercube_array[i]);
                execute(hypercube_array[i]);
            }

            // std::cout << "output_array status: {" << output_array[0] << "} \n";

            if (verbose) {
                record();
                std::printf("The current img category is: %d. ", input_source_idx);
            }

            if (input_source_idx != 10) {
                // The input_source_idx is defacto the same thing as the expected_image_category.
                signal_classification(input_source_idx);
            }
            find_matched_pattern();
            

            // TODO: To make this hypercube an image categorization machine, 
            // we need to first tell the naturally-emerged categoorization fingerprints of each image in training/naming phases
            // before the hypercube can be used in categorization in inference phases.
            //
            // This is an assertion that the image categorization fingerprints can be naturally emerged from this hypercube.
            // Need tests to proove it.
            //
            // Image categorization is just a format of image fingerprints.
            // In theory there is always a way to find the corresponding patterns in this hypercube for images.
            //
            // TODO:
            // Next step is to build an automatic running hypercube verifying this theory, it requires:
            // 1. Lots of different images keep feeding into hypercube.
            // 2. Each image takes around 100 - 1000 runs?
            // 3. Image loading should focus on the center 64 * 64 pixels of the image.
            // 4. Those constant-feeding images will shape internal structures of the hypercube.
            // 5. KEY: Write a new function/algorithm keeping scanning vertexes inside hypercube to build fingerprint patterns
            //    to corresponding categorizations. 
            //    a. Leverage traditional error rate calculations to calculate how accurate the currently found pattern is.
            //    b. Keep optimizing patterns of categorizations for each images to reduce error rate.
            //    c. Each image takes about 100 - 1000 runs to be stable? The pattern is decided after maybe 100 runs.
            //    d. Following AlexNet, let's start with 1000 categorizations. We need to find 1000 different patterns.
            //       The pattern to the same categorization could be different in different runs/tests.
            //    e. Current assumption is that even running hypercube in single-thread CPU can achieve image categorization.
            //       This is because running on multiple-threads/GPU only change the speed, internal structure construction proocess;
            //       but it doesn't change the fundamental relationship construction logic in this hypercube.
            //       So even though it's slow, the same manifold evolving property still takes effects.
        }
    }
}