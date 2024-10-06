#ifndef __KernelGlobal_included
#define __KernelGlobal_included

#include <string>
#include <random>
#include <cstring>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <limits>

#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
// Error checking macro
#define CUDA_CHECK_ERROR(call) {                                      \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                  << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Macro for checking cuBLAS errors
#define CUBLAS_CHECK_ERROR(call)                                           \
    {                                                                      \
        cublasStatus_t err = call;                                         \
        if (err != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << err << std::endl;                         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

#define CUDA_CHECK_ERROR_COLLECTIVE() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
#endif // __KernelGlobal_included