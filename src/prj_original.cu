#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <kernelGlobal.cuh>
#include "prj.cuh"

__global__
void upddenact_kernel(float *axoact, int *Hihjhi, int Hj, int denHi, int Mi, float *denact) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= Hj * denHi)
        return;
    int denNi = denHi * Mi;
    int hj = h / denHi;
    int dhi = h % denHi;
    int hi = Hihjhi[hj * denHi + dhi];
    for (int mi = 0; mi < Mi; mi++)
        denact[hj * denNi + dhi * Mi + mi] = axoact[hi * Mi + mi];
}

void upddenact_cu(float *axoact, int *Hihjhi, int Hj, int denHi, int Mi, float *denact) {
    int blockSize = 32;
    int numBlocks_hjdhi = (Hj * denHi + blockSize - 1) / blockSize;
    upddenact_kernel <<< numBlocks_hjdhi, blockSize>>>(axoact, Hihjhi, Hj, denHi, Mi, denact);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__global__
void updtrcjzp_kernel(float *Xj,
                      int Nj,
                      float fgain, float eps, float tauzjdt, float taupdt,
                      float *Zj, float *Pj) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj >= Nj)
        return;
    Zj[nj] += (fgain * Xj[nj] * (1 - eps) + eps - Zj[nj]) * tauzjdt;
    Pj[nj] += (Zj[nj] - Pj[nj]) * taupdt;
}

__global__
void updtrcizp_kernel(float *Xi,
                      int Hj, int denNi,
                      float fgain, float eps, float tauzidt, float taupdt,
                      float *Zi, float *Pi) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Hj * denNi)
        return;
    int hj = n / denNi;
    int ni = n % denNi;
    int k = hj * denNi + ni;
    Zi[k] += (Xi[k] * fgain * (1 - eps) + eps - Zi[k]) * tauzidt;
    Pi[k] += (Zi[k] - Pi[k]) * taupdt;
}


__global__
void updtrcjip_kernel(float *Zj, float *Zi,
                      int Nj, int Mj, int denNi,
                      float taupdt,
                      float *Pji) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denNi * Nj)
        return;
    int nj = n / denNi;
    int ni = n % denNi;
    int hj = nj / Mj;
    Pji[nj * denNi + ni] += (Zi[hj * denNi + ni] * Zj[nj] - Pji[nj * denNi + ni]) * taupdt;
}


void updtraces_cu(float *denact, float *trgact, float prn,
                  int Hj, int Nj, int Mj, int denNi,
                  float fgain, float eps, float tauzidt, float tauzjdt, float taupdt,
                  float *Zj, float *Zi, float *Pj, float *Pi, float *Pji) {
    float prntaupdt = prn * taupdt;
    int blockSize = 128;
    int numBlocksj = (Nj + blockSize - 1) / blockSize;
    int numBlocksi = (Hj * denNi + blockSize - 1) / blockSize;
    int numBlocksji = (Nj * denNi + blockSize - 1) / blockSize;
    updtrcjzp_kernel <<< numBlocksj, blockSize>>>(trgact, Nj, fgain, eps, tauzjdt, prntaupdt, Zj, Pj);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    updtrcizp_kernel <<< numBlocksi, blockSize>>>(denact, Hj, denNi, fgain, eps, tauzidt, prntaupdt, Zi, Pi);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    updtrcjip_kernel <<< numBlocksji, blockSize>>>(Zj, Zi, Nj, Mj, denNi, prntaupdt, Pji);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__global__
void BCPupdbw_kernel(int Nj, int Mj, int denHi, int denNi, int Mi,
                     float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
                     float eps, float bgain, float wgain, float ewgain, float iwgain) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denNi * Nj)
        return;
    int nj = n / denNi;
    int hj = nj/Mj;
    int dni = n % denNi;
    int k = hj * denNi + dni;
    Bj[nj] = bgain * log(Pj[nj]);
    float wji;
    wji = log(Pji[nj * denNi + dni] / (Pi[k] * Pj[nj]));
    wji *= wgain + (wji > 0) * ewgain + (wji < 0) * iwgain;
    Wji[nj * denNi + dni] = wji;
}

void updbw_cu(int Nj, int Mj, int denHi, int denNi, int Mi,
              float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
              float eps, float bgain, float wgain, float ewgain, float iwgain) {
    int blockSize = 128;
    int numBlocksji = (Nj * denNi + blockSize - 1) / blockSize;
    BCPupdbw_kernel<<< numBlocksji, blockSize>>>(Nj, Mj, denHi, denNi, Mi,
                                                Pj, Pi, Pji, Bj, Wji,
                                                eps, bgain, wgain, ewgain, iwgain);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__global__
void updbwsup_kernel(float *Wji, int Nj, int denNi, float tauzidt,
                     float *bwsupinf, float *bwsup) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj >= Nj)
        return;
    bwsup[nj] += (bwsupinf[nj] - bwsup[nj]) * tauzidt;
}

void updbwsup_cu(float *Zi, float *Bj, float *Wji, int Hj, int Mj, int denNi, float tauzidt,
                 float *bwsupinf, float *bwsup) {
    int Nj = Hj * Mj;
    float alpha = 1.0f, beta = 0.0f;

    // Initialize cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK_ERROR(cublasCreate(&handle));

    // Loop through each hj
    for (int hj = 0; hj < Hj; hj++) {
        // Call cuBLAS gemv for matrix-vector multiplication
        CUBLAS_CHECK_ERROR(cublasSgemv(handle, CUBLAS_OP_T, denNi, Mj, &alpha, 
                                       &Wji[hj * Mj * denNi], denNi, 
                                       &Zi[hj * denNi], 1, &beta, 
                                       &bwsupinf[hj * Mj], 1));
    }

    // Launch the kernel to update bwsup using the GPU
    int blockSize = 256;
    int numBlocksj = (Nj + blockSize - 1) / blockSize;
    updbwsup_kernel<<<numBlocksj, blockSize>>>(Wji, Nj, denNi, tauzidt, bwsupinf, bwsup);

    // Check for CUDA kernel launch errors and synchronize
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Destroy cuBLAS handle
    CUBLAS_CHECK_ERROR(cublasDestroy(handle));
}
