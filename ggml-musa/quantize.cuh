#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE 256

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, musaStream_t stream);

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, musaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, musaStream_t stream);
