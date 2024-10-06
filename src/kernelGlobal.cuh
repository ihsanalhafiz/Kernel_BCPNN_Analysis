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

#include "pop.cuh"
#include "prj.cuh"
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

void updsup_cu(int N, float *lgi, float *bwsup, float *sup, float *supinf, float *act,
               float *ada, float *sada, uint *pnoise,
               float taumdt, float igain, float bwgain, float adgain, float tauadt,
               float sadgain, float tausadt, float nampl, float nfreq);
void updact_cu(int H, int M, float *sup, float *act, float again,
                float *hmax, float *hsum);

void upddenact_cu(float *axoact, int *Hihjhi, int Hj, int denHi, int Mi, float *denact);
void updtraces_cu(float *denact, float *trgact, float prn,
                  int Hj, int Nj, int Mj, int denNi,
                  float fgain, float eps, float tauzidt, float tauzjdt, float taupdt,
                  float *Zj, float *Zi, float *Pj, float *Pi, float *Pji);
void updbw_cu(int Nj, int Mj, int denHi, int denNi, int Mi,
              float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
              float eps, float bgain, float wgain, float ewgain, float iwgain);
void updbwsup_cu(float *Zi, float *Bj, float *Wji, int Hj, int Mj, int denNi, float tauzidt,
                 float *bwsupinf, float *bwsup);

#endif // __KernelGlobal_included