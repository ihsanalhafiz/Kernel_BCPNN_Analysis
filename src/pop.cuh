#ifndef __Pop_included
#define __Pop_included

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <kernelGlobal.cuh>

void updsup_cu(int N, float *lgi, float *bwsup, float *sup, float *supinf, float *act,
               float *ada, float *sada, uint *pnoise,
               float taumdt, float igain, float bwgain, float adgain, float tauadt,
               float sadgain, float tausadt, float nampl, float nfreq);

void updact_cu(int H, int M, float *sup, float *act, float again,
                float *hmax, float *hsum);

#endif // __Pop_included