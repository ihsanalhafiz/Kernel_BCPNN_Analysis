#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <kernelGlobal.cuh>
#include <math_constants.h>
#include "prj.cuh"

__global__
void upddenact_kernel_optimized(const float* __restrict__ axoact,
                                const int* __restrict__ Hihjhi,
                                const int Hj, const int denHi, const int Mi,
                                float* __restrict__ denact) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = Hj * denHi * Mi;

    if (idx >= total_elements)
        return;

    // Compute h and mi from the linear index
    int h = idx / Mi;
    int mi = idx % Mi;

    // Compute hj and dhi from h
    int hj = h / denHi;
    int dhi = h % denHi;

    // Get hi using Hihjhi array
    int hi = Hihjhi[hj * denHi + dhi];

    // Calculate indices for axoact and denact arrays
    int axoact_idx = hi * Mi + mi;
    int denact_idx = hj * denHi * Mi + dhi * Mi + mi;

    // Perform the data copy
    denact[denact_idx] = axoact[axoact_idx];
}

void upddenact_cu_optimized(const float* axoact, const int* Hihjhi,
                  int Hj, int denHi, int Mi, float* denact) {
    int total_elements = Hj * denHi * Mi;
    int blockSize = 128;  // Adjust based on your GPU's characteristics
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    upddenact_kernel_optimized<<<numBlocks, blockSize>>>(
        axoact, Hihjhi, Hj, denHi, Mi, denact);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in upddenact_cu: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

__global__
void updtrcjzp_kernel_optimized(const float* __restrict__ Xj,
                                int Nj,
                                const float fgain, const float eps,
                                const float tauzjdt, const float taupdt,
                                float* __restrict__ Zj,
                                float* __restrict__ Pj) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj >= Nj)
        return;

    // Load values into registers
    float Xj_nj = Xj[nj];
    float Zj_nj = Zj[nj];
    float Pj_nj = Pj[nj];

    // Compute the updates
    float delta_Zj = (fgain * Xj_nj * (1.0f - eps) + eps - Zj_nj) * tauzjdt;
    Zj_nj += delta_Zj;

    float delta_Pj = (Zj_nj - Pj_nj) * taupdt;
    Pj_nj += delta_Pj;

    // Write back to global memory
    Zj[nj] = Zj_nj;
    Pj[nj] = Pj_nj;
}

__global__
void updtrcizp_kernel_optimized(const float* __restrict__ Xi,
                                int Hj, int denNi,
                                const float fgain, const float eps,
                                const float tauzidt, const float taupdt,
                                float* __restrict__ Zi,
                                float* __restrict__ Pi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = Hj * denNi;

    if (idx >= total_elements)
        return;

    // Load values into registers
    float Xi_idx = Xi[idx];
    float Zi_idx = Zi[idx];
    float Pi_idx = Pi[idx];

    // Compute the updates
    float delta_Zi = (Xi_idx * fgain * (1.0f - eps) + eps - Zi_idx) * tauzidt;
    Zi_idx += delta_Zi;

    float delta_Pi = (Zi_idx - Pi_idx) * taupdt;
    Pi_idx += delta_Pi;

    // Write back to global memory
    Zi[idx] = Zi_idx;
    Pi[idx] = Pi_idx;
}

__global__
void updtrcjip_kernel_optimized(const float* __restrict__ Zj,
                                const float* __restrict__ Zi,
                                int Nj, int Mj, int denNi,
                                const float taupdt,
                                float* __restrict__ Pji) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = Nj * denNi;

    if (idx >= total_elements)
        return;

    int nj = idx / denNi;
    int ni = idx % denNi;

    // Compute hj and access Zi accordingly
    int hj = nj / Mj;
    int Zi_idx = hj * denNi + ni;
    int Pji_idx = nj * denNi + ni;

    // Load values into registers
    float Zi_val = Zi[Zi_idx];
    float Zj_val = Zj[nj];
    float Pji_val = Pji[Pji_idx];

    // Compute the update
    float delta_Pji = (Zi_val * Zj_val - Pji_val) * taupdt;
    Pji_val += delta_Pji;

    // Write back to global memory
    Pji[Pji_idx] = Pji_val;
}

void updtraces_cu_optimized(const float* __restrict__ denact,
                  const float* __restrict__ trgact,
                  float prn,
                  int Hj, int Nj, int Mj, int denNi,
                  float fgain, float eps,
                  float tauzidt, float tauzjdt, float taupdt,
                  float* __restrict__ Zj,
                  float* __restrict__ Zi,
                  float* __restrict__ Pj,
                  float* __restrict__ Pi,
                  float* __restrict__ Pji) {
    float prntaupdt = prn * taupdt;
    int blockSize = 128;  // Adjusted block size for better occupancy

    // Kernel 1: updtrcjzp_kernel_optimized
    int numBlocksj = (Nj + blockSize - 1) / blockSize;
    updtrcjzp_kernel_optimized<<<numBlocksj, blockSize>>>(
        trgact, Nj, fgain, eps, tauzjdt, prntaupdt, Zj, Pj);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in updtrcjzp_kernel_optimized: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: updtrcizp_kernel_optimized
    int total_elements_i = Hj * denNi;
    int numBlocksi = (total_elements_i + blockSize - 1) / blockSize;
    updtrcizp_kernel_optimized<<<numBlocksi, blockSize>>>(
        denact, Hj, denNi, fgain, eps, tauzidt, prntaupdt, Zi, Pi);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in updtrcizp_kernel_optimized: %s\n", cudaGetErrorString(err));
    }

    // Kernel 3: updtrcjip_kernel_optimized
    int total_elements_ji = Nj * denNi;
    int numBlocksji = (total_elements_ji + blockSize - 1) / blockSize;
    updtrcjip_kernel_optimized<<<numBlocksji, blockSize>>>(
        Zj, Zi, Nj, Mj, denNi, prntaupdt, Pji);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in updtrcjip_kernel_optimized: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

__global__
void compute_Bj_kernel(int Nj,
                       const float* __restrict__ Pj,
                       float* __restrict__ Bj,
                       const float bgain) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj >= Nj)
        return;

    // Compute Bj[nj]
    Bj[nj] = bgain * logf(Pj[nj]);
}

__global__
void BCPupdbw_kernel_optimized(int Nj, int Mj, int denHi, int denNi, int Mi,
                               const float* __restrict__ Pj,
                               const float* __restrict__ Pi,
                               const float* __restrict__ Pji,
                               const float* __restrict__ Bj,
                               float* __restrict__ Wji,
                               const float eps,
                               const float wgain,
                               const float ewgain,
                               const float iwgain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = Nj * denNi;

    if (idx >= total_elements)
        return;

    int nj = idx / denNi;
    int dni = idx % denNi;
    int hj = nj / Mj;
    int k = hj * denNi + dni;

    // Load values into registers
    float Pj_nj = Pj[nj];
    float Pi_k = Pi[k];
    float Pji_idx = Pji[idx];

    // Compute wji
    float wji = logf(Pji_idx / (Pi_k * Pj_nj));

    // Compute gain without branch divergence
    float pos_mask = (wji > 0.0f);
    float neg_mask = (wji < 0.0f);
    float gain = wgain + ewgain * pos_mask + iwgain * neg_mask;

    // Update wji
    wji *= gain;

    // Write result to global memory
    Wji[idx] = wji;
}

void updbw_cu_optimized(int Nj, int Mj, int denHi, int denNi, int Mi,
              const float* Pj,
              const float* Pi,
              const float* Pji,
              float* Bj,
              float* Wji,
              float eps,
              float bgain,
              float wgain,
              float ewgain,
              float iwgain) {
    // Compute Bj[nj] separately to avoid redundant computations
    int blockSizeBj = 128;
    int numBlocksBj = (Nj + blockSizeBj - 1) / blockSizeBj;

    compute_Bj_kernel<<<numBlocksBj, blockSizeBj>>>(
        Nj, Pj, Bj, bgain);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_Bj_kernel: %s\n", cudaGetErrorString(err));
    }

    // Synchronize before launching the next kernel
    cudaDeviceSynchronize();

    // Launch the optimized BCPupdbw_kernel
    int total_elements = Nj * denNi;
    int blockSize = 256;  // Adjust based on GPU occupancy
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    BCPupdbw_kernel_optimized<<<numBlocks, blockSize>>>(
        Nj, Mj, denHi, denNi, Mi,
        Pj, Pi, Pji, Bj, Wji,
        eps, wgain, ewgain, iwgain);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in BCPupdbw_kernel_optimized: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

__global__
void updbwsup_kernel_optimized(const float* __restrict__ bwsupinf,
                     int Nj,
                     const float tauzidt,
                     float* __restrict__ bwsup) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj >= Nj)
        return;

    float bwsup_nj = bwsup[nj];
    float bwsupinf_nj = bwsupinf[nj];
    bwsup_nj += (bwsupinf_nj - bwsup_nj) * tauzidt;
    bwsup[nj] = bwsup_nj;
}

void updbwsup_cu_optimized(const float* __restrict__ Zi,
                 const float* __restrict__ Bj,
                 const float* __restrict__ Wji,
                 int Hj, int Mj, int denNi, float tauzidt,
                 float* __restrict__ bwsupinf,
                 float* __restrict__ bwsup) {
    int Nj = Hj * Mj;
    float alpha = 1.0f, beta = 0.0f;

    // Initialize cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK_ERROR(cublasCreate(&handle));

    // Allocate host arrays of device pointers
    float** h_Aarray = (float**)malloc(Hj * sizeof(float*));
    float** h_xarray = (float**)malloc(Hj * sizeof(float*));
    float** h_yarray = (float**)malloc(Hj * sizeof(float*));

    // Initialize host arrays with device pointers
    for (int hj = 0; hj < Hj; hj++) {
        h_Aarray[hj] = (float*)(Wji + hj * Mj * denNi);
        h_xarray[hj] = (float*)(Zi + hj * denNi);
        h_yarray[hj] = (float*)(bwsupinf + hj * Mj);
    }

    // Allocate device arrays of pointers
    float** d_Aarray;
    float** d_xarray;
    float** d_yarray;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_Aarray, Hj * sizeof(float*)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_xarray, Hj * sizeof(float*)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_yarray, Hj * sizeof(float*)));

    // Copy host arrays to device arrays
    CUDA_CHECK_ERROR(cudaMemcpy(d_Aarray, h_Aarray, Hj * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_xarray, h_xarray, Hj * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_yarray, h_yarray, Hj * sizeof(float*), cudaMemcpyHostToDevice));

    // Call cublasSgemvBatched
    CUBLAS_CHECK_ERROR(cublasSgemvBatched(
        handle,
        CUBLAS_OP_T,         // Transpose operation
        denNi,               // m
        Mj,                  // n
        &alpha,
        (const float**)d_Aarray, denNi,  // Aarray and leading dimension
        (const float**)d_xarray, 1,      // xarray and increment
        &beta,
        d_yarray, 1,                     // yarray and increment
        Hj));                            // Batch count

    // Free device arrays of pointers
    CUDA_CHECK_ERROR(cudaFree(d_Aarray));
    CUDA_CHECK_ERROR(cudaFree(d_xarray));
    CUDA_CHECK_ERROR(cudaFree(d_yarray));

    // Free host arrays of pointers
    free(h_Aarray);
    free(h_xarray);
    free(h_yarray);

    // Launch the kernel to update bwsup using the GPU
    int blockSize = 128;  // Adjust for better occupancy
    int numBlocksj = (Nj + blockSize - 1) / blockSize;
    updbwsup_kernel_optimized<<<numBlocksj, blockSize>>>(
        bwsupinf, Nj, tauzidt, bwsup);

    // Check for CUDA kernel launch errors and synchronize
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Destroy cuBLAS handle
    CUBLAS_CHECK_ERROR(cublasDestroy(handle));
}
