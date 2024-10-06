#ifndef __Prj_included
#define __Prj_included

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <kernelGlobal.cuh>

void upddenact_cu(float *axoact, int *Hihjhi, int *Chjhi, int Hj, int denHi, int Mi, float *denact);

void upddenact_cu_optimized(const float* axoact, const int* Hihjhi,
                  int Hj, int denHi, int Mi, float* denact);

void updtraces_cu(float *denact, float *trgact, float prn,
                  int Hj, int Nj, int Mj, int denNi,
                  float fgain, float eps, float tauzidt, float tauzjdt, float taupdt,
                  float *Zj, float *Zi, float *Pj, float *Pi, float *Pji);

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
                  float* __restrict__ Pji);

void updbw_cu(int Nj, int Mj, int denHi, int denNi, int Mi,
              float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
              float eps, float bgain, float wgain, float ewgain, float iwgain);

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
              float iwgain);

void updbwsup_cu(float *Zi, float *Bj, float *Wji, int Hj, int Mj, int denNi, float tauzidt,
                 float *bwsupinf, float *bwsup);

void updbwsup_cu_optimized(const float* __restrict__ Zi,
                 const float* __restrict__ Bj,
                 const float* __restrict__ Wji,
                 int Hj, int Mj, int denNi, float tauzidt,
                 float* __restrict__ bwsupinf,
                 float* __restrict__ bwsup);

#endif // __Prj_included