#include "loading_image.h"

#include <iostream>
#include <string>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <jpeglib.h> // Core libjpeg header

#include "constants.h"

/**
 * Loads a JPEG image, resizes/scales it to 64x64, and populates an int32_t array.
 * Pack format: 0xAARRGGBB (Alpha defaults to 0xFF since JPEG has no transparency)
 * * @param filename Path to the JPEG file.
 * @param out_array Reference to the 64x64 destination array.
 * @return true if successful, false otherwise.
 */
bool load_jpeg_to_input_buffer(const std::string& filename, std::vector<int32_t>& out_array) {
    // Open the file using standard C I/O (required by libjpeg)
    FILE* infile = fopen(filename.c_str(), "rb");
    if (!infile) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // Initialize libjpeg error handling and decompression structures
    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    
    // Read the JPEG header info
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        std::cerr << "Error: Failed to read JPEG header." << std::endl;
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }

    // Force libjpeg to convert the image to standard RGB color space
    cinfo.out_color_space = JCS_RGB;

    // --- STEP 1: CALCULATE COARSE SCALE DOWN FOR SPEED ---
    // Calculate final aspect-aware dimensions to see if libjpeg can pre-shrink it
    double scale_w = static_cast<double>(TARGET_WIDTH) / cinfo.image_width;
    double scale_h = static_cast<double>(TARGET_HEIGHT) / cinfo.image_height;
    double target_scale = std::max(scale_w, scale_h); // Ensure shorter side hits 256

    if (target_scale <= 0.125)      { cinfo.scale_num = 1; cinfo.scale_denom = 8; }
    else if (target_scale <= 0.25)  { cinfo.scale_num = 1; cinfo.scale_denom = 4; }
    else if (target_scale <= 0.5)   { cinfo.scale_num = 1; cinfo.scale_denom = 2; }

    // Start decompression
    jpeg_start_decompress(&cinfo);

    int decomp_w = cinfo.output_width;
    int decomp_h = cinfo.output_height;
    int num_channels = cinfo.output_components; // Guaranteed to be 3 (RGB)

    // Allocate a buffer to store the raw decompressed scanlines temporarily
    size_t row_stride = decomp_w * num_channels;
    std::vector<uint8_t> raw_buffer(decomp_h * row_stride);
    
    // Array of row pointers that libjpeg expects
    std::vector<JSAMPROW> row_pointers(decomp_h);
    for (int i = 0; i < decomp_h; ++i) {
        row_pointers[i] = &raw_buffer[i * row_stride];
    }

    // Read all the scanlines out of the file into our raw_buffer
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &row_pointers[cinfo.output_scanline], decomp_h);
    }

    // Finish decompression and clean up libjpeg memory constructs
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    // --- STEP 2: RESIZE SHORTER SIDE TO EXACTLY 256 (ASPECT-PRESERVED) ---
    int resized_w, resized_h;
    if (decomp_w < decomp_h) {
        resized_w = TARGET_WIDTH;
        resized_h = (decomp_h * TARGET_HEIGHT) / decomp_w;
    } else {
        resized_h = TARGET_HEIGHT;
        resized_w = (decomp_w * TARGET_WIDTH) / decomp_h;
    }

    // --- STEP 3: COMPUTE CENTER CROP OFFSETS ---
    int crop_x = (resized_w - TARGET_WIDTH) / 2;
    int crop_y = (resized_h - TARGET_HEIGHT) / 2;

    // --- STEP 4: MAP & PACK DIRECTLY TO TARGET GRID ---
    for (int y = 0; y < TARGET_HEIGHT; ++y) {
        // Map 256-grid coordinates back into the intermediate resized space, factoring in crop offset
        int intermediate_y = y + crop_y;
        // Map intermediate space back to the raw decompressed image buffer
        int src_y = (intermediate_y * decomp_h) / resized_h;
        
        for (int x = 0; x < TARGET_WIDTH; ++x) {
            int intermediate_x = x + crop_x;
            int src_x = (intermediate_x * decomp_w) / resized_w;
            
            size_t pixel_idx = (src_y * row_stride) + (src_x * num_channels);
            
            uint8_t r = raw_buffer[pixel_idx + 0];
            uint8_t g = raw_buffer[pixel_idx + 1];
            uint8_t b = raw_buffer[pixel_idx + 2];
            uint8_t a = 0xFF; // Non-transparent

            // // Pack or set to 1/0 based on your threshold logic
            // int32_t packed_pixel = (a << 24) | (r << 16) | (g << 8) | b;

            // let packed_pixel be either 0 or 1.
            // Standard perceived luminance formula
            uint8_t luminance = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
            // If it's bright, make it 1 (white). If it's dark, make it 0 (black).
            int32_t packed_pixel = (luminance > 127) ? 1 : 0;
            
            // Flatten the 2-dimensional image into one dimensional array.
            out_array[y * TARGET_WIDTH + x] = packed_pixel;
            // out_array[y][x] = packed_pixel;
        }
    }

    // // --- RESAMPLE/NEAREST NEIGHBOR TO EXACT 64x64 GRID ---
    // // Since libjpeg's hardware scaling only works in powers of 2, 
    // // we use a safe, fast nearest-neighbor mapping loop to map to absolute 64x64.
    // for (int y = 0; y < TARGET_HEIGHT; ++y) {
    //     // Map 64-grid coordinate back to the decompressed coordinate space
    //     int src_y = (y * decomp_h) / TARGET_HEIGHT;
        
    //     for (int x = 0; x < TARGET_WIDTH; ++x) {
    //         int src_x = (x * decomp_w) / TARGET_WIDTH;
            
    //         size_t pixel_idx = (src_y * row_stride) + (src_x * num_channels);
            
    //         uint8_t r = raw_buffer[pixel_idx + 0];
    //         uint8_t g = raw_buffer[pixel_idx + 1];
    //         uint8_t b = raw_buffer[pixel_idx + 2];
    //         uint8_t a = 0xFF; // JPEGs do not support transparency data

    //         // Pack channels into standard 32-bit integer (0xAARRGGBB format)
    //         // int32_t packed_pixel = (a << 24) | (r << 16) | (g << 8) | b;
            
    //         // let packed_pixel be either 0 or 1.
    //         // Standard perceived luminance formula
    //         uint8_t luminance = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
    //         // If it's bright, make it 1 (white). If it's dark, make it 0 (black).
    //         int32_t packed_pixel = (luminance > 127) ? 1 : 0;

    //         // Flatten the 2-dimensional image into one dimensional array.
    //         out_array[y * TARGET_WIDTH + x] = packed_pixel;
    //     }
    // }

    return true;
}

// int main() {
//     int32_t pixelGrid[TARGET_HEIGHT * TARGET_WIDTH];
//     std::string path = "photo.jpg";

//     if (load_jpeg_to_input_buffer(path, pixelGrid)) {
//         std::cout << "Successfully decoded and downsampled JPEG into 64x64 grid!" << std::endl;
//         std::cout << "Top-left pixel hex (ARGB): 0x" << std::hex << pixelGrid[0] << std::endl;
//     }
//     return 0;
// }