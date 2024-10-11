#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <float.h>
#include <kernelGlobal.cuh>
#include "pop.cuh"

// Optimized Kernel Function
__global__
void updsup_kernel_optimized(int N,
                             const float *__restrict__ lgi,
                             const float *__restrict__ bwsup,
                             float *__restrict__ sup,
                             float *__restrict__ supinf,
                             const float *__restrict__ act,
                             float *__restrict__ ada,
                             float *__restrict__ sada,
                             const uint *__restrict__ pnoise,
                             const float taumdt,
                             const float igain,
                             const float bwgain,
                             const float adgain,
                             const float tauadt,
                             const float sadgain,
                             const float tausadt,
                             const float nampl,
                             const float nfreq) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N)
        return;

    // Compute supinf_n locally to reduce global memory accesses
    float supinf_n = igain * lgi[n] + bwgain * bwsup[n];
    supinf_n += nampl * (pnoise[n] - nfreq);

    // Update ada[n] without branching
    float ada_n = ada[n] + (adgain * act[n] - ada[n]) * tauadt;
    ada[n] = ada_n;
    supinf_n -= ada_n;

    // Update sada[n] without branching
    float sada_n = sada[n] + (sadgain * act[n] - sada[n]) * tausadt;
    sada[n] = sada_n;
    supinf_n -= sada_n;

    // Update sup[n]
    float sup_n = sup[n] + (supinf_n - sup[n]) * taumdt;
    sup[n] = sup_n;

    // Store the updated supinf_n
    supinf[n] = supinf_n;
}

// Optimized Host Function
void updsup_cu_optimized(int N,
                         float *lgi,
                         float *bwsup,
                         float *sup,
                         float *supinf,
                         float *act,
                         float *ada,
                         float *sada,
                         uint *pnoise,
                         float taumdt,
                         float igain,
                         float bwgain,
                         float adgain,
                         float tauadt,
                         float sadgain,
                         float tausadt,
                         float nampl,
                         float nfreq) {
    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updsup_kernel_optimized, 0, 0);
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the optimized kernel
    updsup_kernel_optimized<<<gridSize, blockSize>>>(
        N, lgi, bwsup, sup, supinf, act, ada, sada, pnoise,
        taumdt, igain, bwgain, adgain, tauadt, sadgain, tausadt, nampl, nfreq);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}



// Optimized Kernel Function
__global__
void fullnorm_kernel_optimized(int H, int M,
                               const float *__restrict__ sup,
                               float *__restrict__ act,
                               const float again,
                               float *__restrict__ hmax,
                               float *__restrict__ hsum) {
    // Each block handles one 'h' (row)
    int h = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[]; // Shared memory for reductions
    float *smax = sdata;             // smax[blockDim.x]
    float *ssum = sdata + blockDim.x; // ssum[blockDim.x]

    int n_start = h * M;
    int n_end = n_start + M;

    // Step 1: Compute local maximum
    float local_max = -FLT_MAX;
    for (int n = n_start + tid; n < n_end; n += blockDim.x) {
        float val = sup[n];
        if (val > local_max)
            local_max = val;
    }

    // Reduction to find global maximum
    smax[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && smax[tid + stride] > smax[tid])
            smax[tid] = smax[tid + stride];
        __syncthreads();
    }

    float hmax_h = smax[0];
    if (tid == 0)
        hmax[h] = hmax_h;
    __syncthreads();

    // Step 2: Compute act[n] and local sum
    float local_sum = 0.0f;
    for (int n = n_start + tid; n < n_end; n += blockDim.x) {
        float exp_val = expf(again * (sup[n] - hmax_h));
        act[n] = exp_val;
        local_sum += exp_val;
    }

    // Reduction to compute global sum
    ssum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }

    float hsum_h = ssum[0];
    if (tid == 0)
        hsum[h] = hsum_h;
    __syncthreads();

    // Step 3: Normalize act[n]
    if (hsum_h > 0.0f) {
        for (int n = n_start + tid; n < n_end; n += blockDim.x) {
            act[n] /= hsum_h;
        }
    }
}

// Optimized Host Function
void updact_cu_optimized(int H, int M,
                         float *sup,
                         float *act,
                         float again,
                         float *hmax,
                         float *hsum) {
    int blockSize;
    int minGridSize;

    // Determine the optimal block size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, fullnorm_kernel_optimized, 0, 0);

    // Ensure blockSize does not exceed M
    blockSize = (blockSize > M) ? M : blockSize;

    int gridSize = H;

    // Calculate shared memory size
    int sharedMemSize = 2 * blockSize * sizeof(float); // smax and ssum

    // Launch the optimized kernel
    fullnorm_kernel_optimized<<<gridSize, blockSize, sharedMemSize>>>(
        H, M, sup, act, again, hmax, hsum);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}
