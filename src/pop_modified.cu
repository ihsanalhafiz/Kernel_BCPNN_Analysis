#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <kernelGlobal.cuh>
#include "pop.cuh"

__global__
void updsup_kernel_optimized(int N,
                             const float* __restrict__ lgi,
                             const float* __restrict__ bwsup,
                             float* __restrict__ sup,
                             float* __restrict__ supinf,
                             const float* __restrict__ act,
                             float* __restrict__ ada,
                             float* __restrict__ sada,
                             const uint* __restrict__ pnoise,
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

    // Load data into registers to minimize global memory access
    float lgi_n = lgi[n];
    float bwsup_n = bwsup[n];
    float sup_n = sup[n];
    float supinf_n = igain * lgi_n + bwgain * bwsup_n;

    // Convert pnoise[n] to float once
    float pnoise_n = static_cast<float>(pnoise[n]);
    supinf_n += nampl * (pnoise_n - nfreq);

    // Declare act_n outside conditional blocks to prevent redundant loads
    float act_n = 0.0f;

    if (adgain != 0.0f) {
        act_n = act[n];  // Load act[n] only if needed
        float ada_n = ada[n];
        ada_n += (adgain * act_n - ada_n) * tauadt;
        ada[n] = ada_n;
        supinf_n -= ada_n;
    }

    if (sadgain != 0.0f) {
        if (adgain == 0.0f) {
            act_n = act[n];  // Load act[n] if it wasn't loaded before
        }
        float sada_n = sada[n];
        sada_n += (sadgain * act_n - sada_n) * tausadt;
        sada[n] = sada_n;
        supinf_n -= sada_n;
    }

    supinf[n] = supinf_n;

    // Update sup[n] with the new value
    sup_n += (supinf_n - sup_n) * taumdt;
    sup[n] = sup_n;
}

void updsup_cu_optimized(int N, float* lgi, float* bwsup, float* sup, float* supinf,
                         float* act, float* ada, float* sada, uint* pnoise, float taumdt,
                         float igain, float bwgain, float adgain, float tauadt, 
                         float sadgain, float tausadt, float nampl, float nfreq) {
    // Adjust blockSize based on your GPU's characteristics
    int blockSize = 128;  // Try 256 or 512 for better occupancy
    int numBlocks = (N + blockSize - 1) / blockSize;

    updsup_kernel_optimized<<<numBlocks, blockSize>>>(
        N, lgi, bwsup, sup, supinf, act, ada, sada, pnoise,
        taumdt, igain, bwgain, adgain, tauadt, sadgain, tausadt,
        nampl, nfreq);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

#include <float.h>

__global__
void fullnorm_kernel_optimized(int H, int M,
                               const float* __restrict__ sup,
                               float* __restrict__ act,
                               const float again,
                               float* __restrict__ hmax,
                               float* __restrict__ hsum) {
    int h = blockIdx.x;  // Each block handles one 'h'
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (h >= H)
        return;

    // Shared memory for reductions
    extern __shared__ float sdata[];

    // Pointers to shared memory
    float* smax = sdata;
    float* ssum = sdata + blockSize;

    // Initialize local maximum
    float local_max = -FLT_MAX;

    // Compute local maximum of sup[n]
    for (int n = tid; n < M; n += blockSize) {
        float sup_n = sup[h * M + n];
        local_max = fmaxf(local_max, sup_n);
    }

    // Store local maximum in shared memory
    smax[tid] = local_max;
    __syncthreads();

    // Reduction to find global maximum
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    float hmax_h;
    if (tid == 0) {
        hmax_h = smax[0];
        hmax[h] = hmax_h;
    }
    __syncthreads();
    hmax_h = hmax[h];  // Broadcast hmax_h to all threads

    // Initialize sum
    float sum = 0.0f;

    // Compute act[n] and partial sums
    for (int n = tid; n < M; n += blockSize) {
        float sup_n = sup[h * M + n];
        float act_n = __expf(again * (sup_n - hmax_h));
        act[h * M + n] = act_n;
        sum += act_n;
    }

    // Store partial sums in shared memory
    ssum[tid] = sum;
    __syncthreads();

    // Reduction to compute total sum
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    float hsum_h;
    if (tid == 0) {
        hsum_h = ssum[0];
        hsum[h] = hsum_h;
    }
    __syncthreads();
    hsum_h = hsum[h];  // Broadcast hsum_h to all threads

    // Normalize act[n] if hsum_h > 0
    if (hsum_h > 0.0f) {
        for (int n = tid; n < M; n += blockSize) {
            act[h * M + n] /= hsum_h;
        }
    }
}

void updact_cu_optimized(int H, int M, float* sup, float* act, float again, float* hmax, float* hsum) {
    int blockSize = 128;  // Adjust based on GPU occupancy
    int numBlocks = H;

    // Calculate shared memory size (2 arrays of blockSize floats)
    size_t sharedMemSize = 2 * blockSize * sizeof(float);

    fullnorm_kernel_optimized<<<numBlocks, blockSize, sharedMemSize>>>(
        H, M, sup, act, again, hmax, hsum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
