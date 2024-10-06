#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <kernelGlobal.cuh>
#include "pop.cuh"

__global__
void updsup_kernel(int N, float *lgi, float *bwsup, float *sup, float *supinf, float *act,
                   float *ada, float *sada, uint *pnoise,
                   float taumdt, float igain, float bwgain, float adgain, float tauadt,
                   float sadgain, float tausadt, float nampl, float nfreq) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (N <= n)
        return;
    supinf[n] = igain * lgi[n] + bwgain * bwsup[n];
    supinf[n] += nampl * (pnoise[n] - nfreq);
    if (adgain != 0) {
        ada[n] += (adgain * act[n] - ada[n]) * tauadt;
        supinf[n] -= ada[n];
    }
    if (sadgain != 0) {
        sada[n] += (sadgain * act[n] - sada[n]) * tausadt;
        supinf[n] -= sada[n];
    }
    sup[n] += (supinf[n] - sup[n]) * taumdt;
}

void updsup_cu(int N, float *lgi, float *bwsup, float *sup, float *supinf, float *act,
               float *ada, float *sada, uint *pnoise,
               float taumdt, float igain, float bwgain, float adgain, float tauadt,
               float sadgain, float tausadt, float nampl, float nfreq) {
    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;
    updsup_kernel <<< numBlocks, blockSize>>>(N, lgi, bwsup, sup, supinf, act, ada, sada, pnoise,
            taumdt, igain, bwgain, adgain, tauadt, sadgain, tausadt,
            nampl, nfreq);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__global__
void fullnorm_kernel(int H, int M, float *sup, float *act, float again,
                     float *hmax, float *hsum) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H)
        return;
    hmax[h] = sup[h * M];
    for (int n = h * M + 1; n < (h + 1) * M; n++)
        if (sup[n] > hmax[h])
            hmax[h] = sup[n];
    hsum[h] = 0;
    for (int n = h * M; n < (h + 1) * M; n++) {
        act[n] = exp(again * (sup[n] - hmax[h]));
        hsum[h] += act[n];
    }
    if (hsum[h] > 0) {
        for (int n = h * M; n < (h + 1) * M; n++)
            act[n] /= hsum[h];
    }
}

void updact_cu(int H, int M, float *sup, float *act, float again,
                float *hmax, float *hsum) {
    int blockSize = 128;
    int numBlocks = H;
    
    fullnorm_kernel <<< numBlocks, blockSize>>>(H, M, sup, act, again, hmax, hsum);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}