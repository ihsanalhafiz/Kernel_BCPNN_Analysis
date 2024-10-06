#ifndef __Prj_included
#define __Prj_included

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <kernelGlobal.cuh>

void upddenact_cu(float *axoact, int *Hihjhi, int *Chjhi, int Hj, int denHi, int Mi, float *denact);
void updtraces_cu(float *denact, float *trgact, float prn,
                  int Hj, int Nj, int Mj, int denNi,
                  float fgain, float eps, float tauzidt, float tauzjdt, float taupdt,
                  float *Zj, float *Zi, float *Pj, float *Pi, float *Pji);
void updbw_cu(int Nj, int Mj, int denHi, int denNi, int Mi,
              float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
              float eps, float bgain, float wgain, float ewgain, float iwgain);
void updbwsup_cu(float *Zi, float *Bj, float *Wji, int Hj, int Mj, int denNi, float tauzidt,
                 float *bwsupinf, float *bwsup);

#endif // __Prj_included