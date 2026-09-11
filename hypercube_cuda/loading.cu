%%writefile loading.cu

#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvjpeg.h>

// Error-checking helper macro for nvJPEG API
#define CHECK_NVJPEG(call) { \
    nvjpegStatus_t status = call; \
    if (status != NVJPEG_STATUS_SUCCESS) { \
        std::cerr << "nvJPEG Error at line " << __LINE__ << ": " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Error-checking helper macro for CUDA API
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int load_image(std::string image_path, uint8_t*& d_raw_rgb, size_t& total_elements) {
  // 1. Read Compressed JPEG File Bytes into PINNED Host Memory
    std::ifstream file(image_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open image file: " << image_path << std::endl;
        return -1;
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    unsigned char* h_jpeg_bitstream = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_jpeg_bitstream, file_size));

    if (!file.read((char*)h_jpeg_bitstream, file_size)) {
        std::cerr << "Failed to read JPEG data from disk." << std::endl;
        cudaFreeHost(h_jpeg_bitstream);
        return -1;
    }
    file.close();

    // 2. Initialize nvJPEG Handles
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    CHECK_NVJPEG(nvjpegCreateSimple(&nvjpeg_handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));

    int num_components = 0;
    nvjpegChromaSubsampling_t subsampling;
    int img_w[NVJPEG_MAX_COMPONENT] = {0};
    int img_h[NVJPEG_MAX_COMPONENT] = {0};

    CHECK_NVJPEG(nvjpegGetImageInfo(
        nvjpeg_handle,
        h_jpeg_bitstream, file_size,
        &num_components, &subsampling,
        img_w, img_h
    ));

    int width = img_w[0];
    int height = img_h[0];

    std::cout << "[nvJPEG] Header verified: " << width << "x" << height
              << ", Channels: " << num_components << std::endl;

    // 3. Allocate Planar VRAM Buffers
    size_t plane_size = width * height;           // Elements per channel
    total_elements = plane_size * 3;       // Total uint8_t elements
    size_t raw_rgb_bytes = total_elements * sizeof(uint8_t);  // Total bytes in VRAM

    std::cout << "Number of elements: " << total_elements << std::endl;
    std::cout << "Allocation size: " << raw_rgb_bytes << " bytes" << std::endl;

    // uint8_t* d_raw_rgb = nullptr;
    CHECK_CUDA(cudaMalloc(&d_raw_rgb, raw_rgb_bytes));

    nvjpegImage_t nv_img;
    nv_img.channel[0] = d_raw_rgb;                   // Red
    nv_img.channel[1] = d_raw_rgb + plane_size;       // Green
    nv_img.channel[2] = d_raw_rgb + (2 * plane_size); // Blue
    nv_img.pitch[0]   = width;
    nv_img.pitch[1]   = width;
    nv_img.pitch[2]   = width;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 4. Execute Hardware Decompression on GPU Stream
    CHECK_NVJPEG(nvjpegDecode(
        nvjpeg_handle, nvjpeg_state,
        h_jpeg_bitstream, file_size,
        NVJPEG_OUTPUT_RGB, &nv_img, stream
    ));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "[nvJPEG] Hardware JPEG decoding complete on GPU." << std::endl;

    cudaDeviceSynchronize();

    // FREE EARLY: Unpin and return host memory IMMEDIATELY
    cudaFreeHost(h_jpeg_bitstream);

    return 0;
}