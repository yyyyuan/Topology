#include "loading_image.h"

#include <iostream>
#include <string>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <jpeglib.h> // Core libjpeg header

const int TARGET_WIDTH = 64;
const int TARGET_HEIGHT = 64;

/**
 * Loads a JPEG image, resizes/scales it to 64x64, and populates an int32_t array.
 * Pack format: 0xAARRGGBB (Alpha defaults to 0xFF since JPEG has no transparency)
 * * @param filename Path to the JPEG file.
 * @param out_array Reference to the 64x64 destination array.
 * @return true if successful, false otherwise.
 */
bool loadJpegTo64x64Array(const std::string& filename, std::vector<int32_t>& out_array) {
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

    // --- LIBJPEG SCALE FACTOR OPTIMIZATION ---
    // libjpeg can hardware-accelerate downsampling during decompression (1/2, 1/4, 1/8 size).
    // Let's compute a mathematically sound scale factor based on our 64x64 requirement.
    if (cinfo.image_width >= TARGET_WIDTH * 8 && cinfo.image_height >= TARGET_HEIGHT * 8) {
        cinfo.scale_num = 1; cinfo.scale_denom = 8;
    } else if (cinfo.image_width >= TARGET_WIDTH * 4 && cinfo.image_height >= TARGET_HEIGHT * 4) {
        cinfo.scale_num = 1; cinfo.scale_denom = 4;
    } else if (cinfo.image_width >= TARGET_WIDTH * 2 && cinfo.image_height >= TARGET_HEIGHT * 2) {
        cinfo.scale_num = 1; cinfo.scale_denom = 2;
    }

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

    // --- RESAMPLE/NEAREST NEIGHBOR TO EXACT 64x64 GRID ---
    // Since libjpeg's hardware scaling only works in powers of 2, 
    // we use a safe, fast nearest-neighbor mapping loop to map to absolute 64x64.
    for (int y = 0; y < TARGET_HEIGHT; ++y) {
        // Map 64-grid coordinate back to the decompressed coordinate space
        int src_y = (y * decomp_h) / TARGET_HEIGHT;
        
        for (int x = 0; x < TARGET_WIDTH; ++x) {
            int src_x = (x * decomp_w) / TARGET_WIDTH;
            
            size_t pixel_idx = (src_y * row_stride) + (src_x * num_channels);
            
            uint8_t r = raw_buffer[pixel_idx + 0];
            uint8_t g = raw_buffer[pixel_idx + 1];
            uint8_t b = raw_buffer[pixel_idx + 2];
            uint8_t a = 0xFF; // JPEGs do not support transparency data

            // Pack channels into standard 32-bit integer (0xAARRGGBB format)
            // int32_t packed_pixel = (a << 24) | (r << 16) | (g << 8) | b;
            
            // let packed_pixel be either 0 or 1.
            // Standard perceived luminance formula
            uint8_t luminance = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
            // If it's bright, make it 1 (white). If it's dark, make it 0 (black).
            int32_t packed_pixel = (luminance > 127) ? 1 : 0;

            // Flatten the 2-dimensional image into one dimensional array.
            out_array[y * TARGET_WIDTH + x] = packed_pixel;
        }
    }

    return true;
}

// int main() {
//     int32_t pixelGrid[TARGET_HEIGHT * TARGET_WIDTH];
//     std::string path = "photo.jpg";

//     if (loadJpegTo64x64Array(path, pixelGrid)) {
//         std::cout << "Successfully decoded and downsampled JPEG into 64x64 grid!" << std::endl;
//         std::cout << "Top-left pixel hex (ARGB): 0x" << std::hex << pixelGrid[0] << std::endl;
//     }
//     return 0;
// }