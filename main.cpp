#include <vector>
#include <cstring>
#include <string>
#include <random>
#include <cstring>
#include <time.h>
#include <sys/time.h>
#include <algorithm> // shuffle
#include <iostream>
#include <limits>
#include "kernelGlobal.cuh"

const int NumberPop = 3;
const float eps_hls = 1e-8;
const float eps_neg_hls = -1e-8;
const float EPS_GLB = 1e-7;

// Layer Population Input
const int H_in = 784;
const int M_in = 2;
const int N_in = H_in * M_in;

// Layer Population Hidden
const int H_hid = 256;
const int M_hid = 512;
const int N_hid = H_hid * M_hid;
const float M_hid_inv = 1.0f/M_hid;
const int log2M_hid = 7;

// Layer Population Output
const int H_ut = 1;
const int M_ut = 10;
const int N_ut = H_ut * M_ut;

const int nactHi_pop = 128;
const int nsilHi_pop = 0;
const float fgain = 1.0;
const float tauzjdt = 1.0;
const float tauzjdt_neg = 0.0;
const float tauzidt = 1.0;
const float again_hls = 1.0;

// Layer Projection Input to Hidden
const int axoHi_ih = H_in;
const int axoNi_ih = axoHi_ih*M_in;
const int nactHi_ih = nactHi_pop;
const int denHi_ih = nactHi_ih + nsilHi_pop;
const int denNi_ih = denHi_ih*M_in;
const int denNi_ih_log2 = 8;

// Layer Projection Hidden to Output
const int axoHi_hu = H_hid;
const int axoNi_hu = axoHi_hu*M_hid;
const int nactHi_hu = axoHi_hu;
const int denHi_hu = nactHi_hu + nsilHi_pop;
const int denNi_hu = denHi_hu*M_hid;

const float eps_hls_m_tauzjdt = eps_hls*tauzjdt;
const float eps_hls_m_tauzjdt_neg = -eps_hls_m_tauzjdt;
const float eps_hls_m_tauzidt = eps_hls*tauzidt;
const float eps_hls_m_tauzidt_neg = -eps_hls_m_tauzidt;

const float tauzidt_neg = (1-tauzidt);

const float taumdt = 1.0;
const float igain = 1.0;
const float bwgain = 1.0;
const float adgain = 1.0;
const float tauadt = 1.0;
const float sadgain = 1.0;
const float tausadt = 1.0;
const float nampl = 1.0;
const float nfreq = 0.1;
const float taupdt = 1.0;
const float bgain = 1.0;
const float wgain = 1.0; 
const float ewgain = 1.0; 
const float iwgain = 1.0;

const float prn = 0.1;

// main function
int main() {
    // Initialize arrays
    float* lgi_hid = new float[N_hid];
    float* bwsup_hid = new float[N_hid];
    float* supinf_hid = new float[N_hid];
    float* sup_hid = new float[N_hid];
    float* act_hid = new float[N_hid];
    uint32_t* pnoise_hid = new uint32_t[N_hid];
    float* rndPoisson_hid = new float[N_hid];

    float* ada_hid = new float[N_hid];
    float* sada_hid = new float[N_hid];

    float* axoact_ih = new float[N_in];
    float* denact_ih = new float[H_hid * denNi_ih];
    int* Hihjhi_ih = new int[H_hid * denHi_ih];
    float* trgpopact_ih = new float[N_hid];
    float* bwsup_ih = new float[N_hid];
    float* bwsupinf_ih = new float[N_hid];

    float* Zj_ih = new float[N_hid];
    float* Zi_ih = new float[H_hid * denNi_ih];
    float* Pj_ih = new float[N_hid];
    float* Pi_ih = new float[H_hid * denNi_ih];
    float* Pji_ih = new float[N_hid * denNi_ih];
    float* Bj_ih = new float[N_hid];
    float* Wji_ih = new float[N_hid * denNi_ih];

    // initialize same arrays for optimized version
    float* lgi_hid_opt = new float[N_hid];
    float* bwsup_hid_opt = new float[N_hid];
    float* supinf_hid_opt = new float[N_hid];
    float* sup_hid_opt = new float[N_hid];
    float* act_hid_opt = new float[N_hid];
    uint32_t* pnoise_hid_opt = new uint32_t[N_hid];
    float* rndPoisson_hid_opt = new float[N_hid];

    float* ada_hid_opt = new float[N_hid];
    float* sada_hid_opt = new float[N_hid];

    float* axoact_ih_opt = new float[N_in];
    float* denact_ih_opt = new float[H_hid * denNi_ih];
    int* Hihjhi_ih_opt = new int[H_hid * denHi_ih];
    float* trgpopact_ih_opt = new float[N_hid];
    float* bwsup_ih_opt = new float[N_hid];
    float* bwsupinf_ih_opt = new float[N_hid];

    float* Zj_ih_opt = new float[N_hid];
    float* Zi_ih_opt = new float[H_hid * denNi_ih];
    float* Pj_ih_opt = new float[N_hid];
    float* Pi_ih_opt = new float[H_hid * denNi_ih];
    float* Pji_ih_opt = new float[N_hid * denNi_ih];
    float* Bj_ih_opt = new float[N_hid];
    float* Wji_ih_opt = new float[N_hid * denNi_ih];

    // Seed for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Print initilize status
    std::cout << "Initialize data fot testing ..." << std::endl;

    // Assign random values to arrays
    for (int i = 0; i < N_hid; ++i) {
        lgi_hid[i] = dis(gen);
        bwsup_hid[i] = dis(gen);
        supinf_hid[i] = dis(gen);
        sup_hid[i] = dis(gen);
        act_hid[i] = dis(gen);
        pnoise_hid[i] = static_cast<uint32_t>(dis(gen) * std::numeric_limits<uint32_t>::max());
        rndPoisson_hid[i] = dis(gen);
        ada_hid[i] = dis(gen);
        sada_hid[i] = dis(gen);
        trgpopact_ih[i] = dis(gen);
        bwsup_ih[i] = dis(gen);
        bwsupinf_ih[i] = dis(gen);
        Zj_ih[i] = dis(gen);
        Pj_ih[i] = dis(gen);
        Bj_ih[i] = dis(gen);
    }

    for (int i = 0; i < N_in; ++i) {
        axoact_ih[i] = dis(gen);
    }

    for (int i = 0; i < H_hid * denNi_ih; ++i) {
        denact_ih[i] = dis(gen);
        Zi_ih[i] = dis(gen);
        Pi_ih[i] = dis(gen);
        Pji_ih[i] = dis(gen);
        Wji_ih[i] = dis(gen);
    }

    for (int i = 0; i < H_hid * denHi_ih; ++i) {
        Hihjhi_ih[i] = i%2;
    }

    // Allocate memory on GPU
    float *d_lgi_hid, *d_bwsup_hid, *d_supinf_hid, *d_sup_hid, *d_act_hid, *d_rndPoisson_hid;
    float *d_ada_hid, *d_sada_hid, *d_axoact_ih, *d_denact_ih, *d_trgpopact_ih, *d_bwsup_ih, *d_bwsupinf_ih;
    float *d_Zj_ih, *d_Zi_ih, *d_Pj_ih, *d_Pi_ih, *d_Pji_ih, *d_Bj_ih, *d_Wji_ih;
    float *d_hmax, *d_hsum;
    uint32_t *d_pnoise_hid;
    int *d_Hihjhi_ih;

    float *d_lgi_hid_opt, *d_bwsup_hid_opt, *d_supinf_hid_opt, *d_sup_hid_opt, *d_act_hid_opt, *d_rndPoisson_hid_opt;
    float *d_ada_hid_opt, *d_sada_hid_opt, *d_axoact_ih_opt, *d_denact_ih_opt, *d_trgpopact_ih_opt, *d_bwsup_ih_opt, *d_bwsupinf_ih_opt;
    float *d_Zj_ih_opt, *d_Zi_ih_opt, *d_Pj_ih_opt, *d_Pi_ih_opt, *d_Pji_ih_opt, *d_Bj_ih_opt, *d_Wji_ih_opt;
    float *d_hmax_opt, *d_hsum_opt;
    uint32_t *d_pnoise_hid_opt;
    int *d_Hihjhi_ih_opt;

    //print allocate memory on GPU
    std::cout << "Allocate memory on GPU ..." << std::endl;

    cudaMalloc((void**)&d_lgi_hid, N_hid * sizeof(float));
    cudaMalloc((void**)&d_bwsup_hid, N_hid * sizeof(float));
    cudaMalloc((void**)&d_supinf_hid, N_hid * sizeof(float));
    cudaMalloc((void**)&d_sup_hid, N_hid * sizeof(float));
    cudaMalloc((void**)&d_act_hid, N_hid * sizeof(float));
    cudaMalloc((void**)&d_pnoise_hid, N_hid * sizeof(uint32_t));
    cudaMalloc((void**)&d_rndPoisson_hid, N_hid * sizeof(float));
    cudaMalloc((void**)&d_ada_hid, N_hid * sizeof(float));
    cudaMalloc((void**)&d_sada_hid, N_hid * sizeof(float));
    cudaMalloc((void**)&d_axoact_ih, N_in * sizeof(float));
    cudaMalloc((void**)&d_denact_ih, H_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_Hihjhi_ih, H_hid * denHi_ih * sizeof(int));
    cudaMalloc((void**)&d_trgpopact_ih, N_hid * sizeof(float));
    cudaMalloc((void**)&d_bwsup_ih, N_hid * sizeof(float));
    cudaMalloc((void**)&d_bwsupinf_ih, N_hid * sizeof(float));
    cudaMalloc((void**)&d_Zj_ih, N_hid * sizeof(float));
    cudaMalloc((void**)&d_Zi_ih, H_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_Pj_ih, N_hid * sizeof(float));
    cudaMalloc((void**)&d_Pi_ih, H_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_Pji_ih, N_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_Bj_ih, N_hid * sizeof(float));
    cudaMalloc((void**)&d_Wji_ih, N_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_hmax, H_hid * sizeof(float));
    cudaMalloc((void**)&d_hsum, H_hid * sizeof(float));

    cudaMalloc((void**)&d_lgi_hid_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_bwsup_hid_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_supinf_hid_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_sup_hid_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_act_hid_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_pnoise_hid_opt, N_hid * sizeof(uint32_t));
    cudaMalloc((void**)&d_rndPoisson_hid_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_ada_hid_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_sada_hid_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_axoact_ih_opt, N_in * sizeof(float));
    cudaMalloc((void**)&d_denact_ih_opt, H_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_Hihjhi_ih_opt, H_hid * denHi_ih * sizeof(int));
    cudaMalloc((void**)&d_trgpopact_ih_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_bwsup_ih_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_bwsupinf_ih_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_Zj_ih_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_Zi_ih_opt, H_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_Pj_ih_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_Pi_ih_opt, H_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_Pji_ih_opt, N_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_Bj_ih_opt, N_hid * sizeof(float));
    cudaMalloc((void**)&d_Wji_ih_opt, N_hid * denNi_ih * sizeof(float));
    cudaMalloc((void**)&d_hmax_opt, H_hid * sizeof(float));
    cudaMalloc((void**)&d_hsum_opt, H_hid * sizeof(float));

    CUDA_CHECK_ERROR_COLLECTIVE();

    std::cout << "Copy data from host to device ..." << std::endl;

    // Copy data from host to device
    cudaMemcpy(d_lgi_hid, lgi_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bwsup_hid, bwsup_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_supinf_hid, supinf_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sup_hid, sup_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act_hid, act_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pnoise_hid, pnoise_hid, N_hid * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rndPoisson_hid, rndPoisson_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ada_hid, ada_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sada_hid, sada_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_axoact_ih, axoact_ih, N_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_denact_ih, denact_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Hihjhi_ih, Hihjhi_ih, H_hid * denHi_ih * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trgpopact_ih, trgpopact_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bwsup_ih, bwsup_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bwsupinf_ih, bwsupinf_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Zj_ih, Zj_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Zi_ih, Zi_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pj_ih, Pj_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pi_ih, Pi_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pji_ih, Pji_ih, N_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bj_ih, Bj_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wji_ih, Wji_ih, N_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_lgi_hid_opt, lgi_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bwsup_hid_opt, bwsup_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_supinf_hid_opt, supinf_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sup_hid_opt, sup_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act_hid_opt, act_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pnoise_hid_opt, pnoise_hid, N_hid * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rndPoisson_hid_opt, rndPoisson_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ada_hid_opt, ada_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sada_hid_opt, sada_hid, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_axoact_ih_opt, axoact_ih, N_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_denact_ih_opt, denact_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Hihjhi_ih_opt, Hihjhi_ih, H_hid * denHi_ih * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trgpopact_ih_opt, trgpopact_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bwsup_ih_opt, bwsup_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bwsupinf_ih_opt, bwsupinf_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Zj_ih_opt, Zj_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Zi_ih_opt, Zi_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pj_ih_opt, Pj_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pi_ih_opt, Pi_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pji_ih_opt, Pji_ih, N_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bj_ih_opt, Bj_ih, N_hid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wji_ih_opt, Wji_ih, N_hid * denNi_ih * sizeof(float), cudaMemcpyHostToDevice);

    CUDA_CHECK_ERROR_COLLECTIVE();

    std::cout << "Run kernel on Population..." << std::endl;
    
    updsup_cu(N_hid, d_lgi_hid, d_bwsup_hid, d_sup_hid, d_supinf_hid, d_act_hid, 
              d_ada_hid, d_sada_hid, d_pnoise_hid, taumdt, igain, bwgain, adgain, tauadt, sadgain, 
                tausadt, nampl, nfreq);

    updsup_cu_optimized(N_hid, d_lgi_hid_opt, d_bwsup_hid_opt, d_sup_hid_opt, d_supinf_hid_opt, d_act_hid_opt, 
              d_ada_hid_opt, d_sada_hid_opt, d_pnoise_hid_opt, taumdt, igain, bwgain, adgain, tauadt, sadgain, 
                tausadt, nampl, nfreq);

    updact_cu(H_hid, M_hid, d_sup_hid, d_act_hid, again_hls, d_hmax, d_hsum);

    updact_cu_optimized(H_hid, M_hid, d_sup_hid_opt, d_act_hid_opt, again_hls, d_hmax_opt, d_hsum_opt);

    std::cout << "Run kernel on Projection..." << std::endl;

    updtraces_cu(d_denact_ih, d_trgpopact_ih, prn, H_hid, N_hid, M_hid, denNi_ih, fgain, eps_hls, 
                tauzidt, tauzjdt, taupdt, d_Zj_ih, d_Zi_ih, d_Pj_ih, d_Pi_ih, d_Pji_ih);
    
    updtraces_cu_optimized(d_denact_ih_opt, d_trgpopact_ih_opt, prn, H_hid, N_hid, M_hid, denNi_ih, fgain, eps_hls,
                tauzidt, tauzjdt, taupdt, d_Zj_ih_opt, d_Zi_ih_opt, d_Pj_ih_opt, d_Pi_ih_opt, d_Pji_ih_opt);

    updbw_cu(N_hid, M_hid, denHi_ih, denNi_ih, M_in, d_Pj_ih, d_Pi_ih, d_Pji_ih, d_Bj_ih, d_Wji_ih,
            eps_hls, bgain, wgain, ewgain, iwgain);
    
    updbw_cu_optimized(N_hid, M_hid, denHi_ih, denNi_ih, M_in, d_Pj_ih_opt, d_Pi_ih_opt, d_Pji_ih_opt, d_Bj_ih_opt, d_Wji_ih_opt,
            eps_hls, bgain, wgain, ewgain, iwgain);

    updbwsup_cu(d_Zi_ih, d_Bj_ih, d_Wji_ih, H_hid, M_hid, denNi_ih, tauzidt, d_bwsupinf_ih, d_bwsup_ih);

    updbwsup_cu_optimized(d_Zi_ih_opt, d_Bj_ih_opt, d_Wji_ih_opt, H_hid, M_hid, denNi_ih, tauzidt, d_bwsupinf_ih_opt, d_bwsup_ih_opt);

    // Copy data from device to host
    cudaMemcpy(lgi_hid, d_lgi_hid, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bwsup_hid, d_bwsup_hid, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(supinf_hid, d_supinf_hid, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sup_hid, d_sup_hid, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(act_hid, d_act_hid, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pnoise_hid, d_pnoise_hid, N_hid * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(rndPoisson_hid, d_rndPoisson_hid, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ada_hid, d_ada_hid, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sada_hid, d_sada_hid, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(axoact_ih, d_axoact_ih, N_in * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(denact_ih, d_denact_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Hihjhi_ih, d_Hihjhi_ih, H_hid * denHi_ih * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(trgpopact_ih, d_trgpopact_ih, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bwsup_ih, d_bwsup_ih, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bwsupinf_ih, d_bwsupinf_ih, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Zj_ih, d_Zj_ih, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Zi_ih, d_Zi_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pj_ih, d_Pj_ih, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pi_ih, d_Pi_ih, H_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pji_ih, d_Pji_ih, N_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bj_ih, d_Bj_ih, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Wji_ih, d_Wji_ih, N_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy data from device to host for optimized version
    cudaMemcpy(lgi_hid_opt, d_lgi_hid_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bwsup_hid_opt, d_bwsup_hid_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(supinf_hid_opt, d_supinf_hid_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sup_hid_opt, d_sup_hid_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(act_hid_opt, d_act_hid_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pnoise_hid_opt, d_pnoise_hid_opt, N_hid * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(rndPoisson_hid_opt, d_rndPoisson_hid_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ada_hid_opt, d_ada_hid_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sada_hid_opt, d_sada_hid_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(axoact_ih_opt, d_axoact_ih_opt, N_in * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(denact_ih_opt, d_denact_ih_opt, H_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Hihjhi_ih_opt, d_Hihjhi_ih_opt, H_hid * denHi_ih * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(trgpopact_ih_opt, d_trgpopact_ih_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bwsup_ih_opt, d_bwsup_ih_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bwsupinf_ih_opt, d_bwsupinf_ih_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Zj_ih_opt, d_Zj_ih_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Zi_ih_opt, d_Zi_ih_opt, H_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pj_ih_opt, d_Pj_ih_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pi_ih_opt, d_Pi_ih_opt, H_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pji_ih_opt, d_Pji_ih_opt, N_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bj_ih_opt, d_Bj_ih_opt, N_hid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Wji_ih_opt, d_Wji_ih_opt, N_hid * denNi_ih * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors
    CUDA_CHECK_ERROR_COLLECTIVE();

    // Compare variable sup_hid and sup_hid_opt for checking correctness updsup kernel
    float diff_sup_hid = 0.0;
    for (int i = 0; i < N_hid; ++i) {
        diff_sup_hid += std::abs(sup_hid[i] - sup_hid_opt[i]);
    }
    std::cout << "Check correctness of updsup kernel ..." << std::endl;
    std::cout << "Difference between sup_hid and sup_hid_opt: " << diff_sup_hid << std::endl;

    // Compare variable act_hid and act_hid_opt for checking correctness updact kernel
    float diff_act_hid = 0.0;
    for (int i = 0; i < N_hid; ++i) {
        diff_act_hid += std::abs(act_hid[i] - act_hid_opt[i]);
    }
    std::cout << "Check correctness of updact kernel ..." << std::endl;
    std::cout << "Difference between act_hid and act_hid_opt: " << diff_act_hid << std::endl;

    // Compare variable Zj_ih and Zj_ih_opt for checking correctness updtraces kernel
    // Compare variable Zi_ih and Zi_ih_opt for checking correctness updtraces kernel
    // Compare variable Pj_ih and Pj_ih_opt for checking correctness updtraces kernel
    // Compare variable Pi_ih and Pi_ih_opt for checking correctness updtraces kernel
    // Compare variable Pji_ih and Pji_ih_opt for checking correctness updtraces kernel
    float diff_Zj_ih = 0.0;
    float diff_Zi_ih = 0.0;
    float diff_Pj_ih = 0.0;
    float diff_Pi_ih = 0.0;
    float diff_Pji_ih = 0.0;
    for (int i = 0; i < N_hid; ++i) {
        diff_Zj_ih += std::abs(Zj_ih[i] - Zj_ih_opt[i]);
        diff_Pj_ih += std::abs(Pj_ih[i] - Pj_ih_opt[i]);
    }
    for (int i = 0; i < H_hid * denNi_ih; ++i) {
        diff_Zi_ih += std::abs(Zi_ih[i] - Zi_ih_opt[i]);
        diff_Pi_ih += std::abs(Pi_ih[i] - Pi_ih_opt[i]);
    }
    for (int i = 0; i < N_hid * denNi_ih; ++i) {
        diff_Pji_ih += std::abs(Pji_ih[i] - Pji_ih_opt[i]);
    }
    std::cout << "Check correctness of updtraces kernel ..." << std::endl;
    std::cout << "Difference between Zj_ih and Zj_ih_opt: " << diff_Zj_ih << std::endl;
    std::cout << "Difference between Zi_ih and Zi_ih_opt: " << diff_Zi_ih << std::endl;
    std::cout << "Difference between Pj_ih and Pj_ih_opt: " << diff_Pj_ih << std::endl;
    std::cout << "Difference between Pi_ih and Pi_ih_opt: " << diff_Pi_ih << std::endl;
    std::cout << "Difference between Pji_ih and Pji_ih_opt: " << diff_Pji_ih << std::endl;

    // Compare variable Bj_ih and Bj_ih_opt for checking correctness updbw kernel
    // Compare variable Wji_ih and Wji_ih_opt for checking correctness updbw kernel
    float diff_Bj_ih = 0.0;
    float diff_Wji_ih = 0.0;
    for (int i = 0; i < N_hid; ++i) {
        diff_Bj_ih += std::abs(Bj_ih[i] - Bj_ih_opt[i]);
    }
    for (int i = 0; i < N_hid * denNi_ih; ++i) {
        diff_Wji_ih += std::abs(Wji_ih[i] - Wji_ih_opt[i]);
    }
    std::cout << "Check correctness of updbw kernel ..." << std::endl;
    std::cout << "Difference between Bj_ih and Bj_ih_opt: " << diff_Bj_ih << std::endl;
    std::cout << "Difference between Wji_ih and Wji_ih_opt: " << diff_Wji_ih << std::endl;

    // Compare variable bwsup_ih and bwsup_ih_opt for checking correctness updbwsup kernel
    float diff_bwsup_ih = 0.0;
    for (int i = 0; i < N_hid; ++i) {
        diff_bwsup_ih += std::abs(bwsup_ih[i] - bwsup_ih_opt[i]);
    }
    std::cout << "Check correctness of updbwsup kernel ..." << std::endl;
    std::cout << "Difference between bwsup_ih and bwsup_ih_opt: " << diff_bwsup_ih << std::endl;

    // verify the correctness of the kernel for all the variables
    if (diff_sup_hid < 1e-6 && diff_act_hid < 1e-6 && diff_Zj_ih < 1e-6 && diff_Zi_ih < 1e-6 && 
        diff_Pj_ih < 1e-6 && diff_Pi_ih < 1e-6 && diff_Pji_ih < 1e-6 && diff_Bj_ih < 1e-6 && diff_Wji_ih < 1e-6 && diff_bwsup_ih < 1e-1) {
        std::cout << "All kernels are correct. Optimized and original kernels have similar results" << std::endl;
    } else {
        std::cout << "There is an error in the kernels." << std::endl;
    }

    std::cout << "Free memory on Host ..." << std::endl;
    // Deallocate host memory
    delete[] lgi_hid;
    delete[] bwsup_hid;
    delete[] supinf_hid;
    delete[] sup_hid;
    delete[] act_hid;
    delete[] pnoise_hid;
    delete[] rndPoisson_hid;
    delete[] ada_hid;
    delete[] sada_hid;
    delete[] axoact_ih;
    delete[] denact_ih;
    delete[] Hihjhi_ih;
    delete[] trgpopact_ih;
    delete[] bwsup_ih;
    delete[] bwsupinf_ih;
    delete[] Zj_ih;
    delete[] Zi_ih;
    delete[] Pj_ih;
    delete[] Pi_ih;
    delete[] Pji_ih;
    delete[] Bj_ih;
    delete[] Wji_ih;

    delete[] lgi_hid_opt;
    delete[] bwsup_hid_opt;
    delete[] supinf_hid_opt;
    delete[] sup_hid_opt;
    delete[] act_hid_opt;
    delete[] pnoise_hid_opt;
    delete[] rndPoisson_hid_opt;
    delete[] ada_hid_opt;
    delete[] sada_hid_opt;
    delete[] axoact_ih_opt;
    delete[] denact_ih_opt;
    delete[] Hihjhi_ih_opt;
    delete[] trgpopact_ih_opt;
    delete[] bwsup_ih_opt;
    delete[] bwsupinf_ih_opt;
    delete[] Zj_ih_opt;
    delete[] Zi_ih_opt;
    delete[] Pj_ih_opt;
    delete[] Pi_ih_opt;
    delete[] Pji_ih_opt;
    delete[] Bj_ih_opt;
    delete[] Wji_ih_opt;

    std::cout << "Free memory on GPU ..." << std::endl;

    cudaDeviceSynchronize();

    // Deallocate GPU memory
    CUDA_CHECK_ERROR(cudaFree(d_lgi_hid));
    CUDA_CHECK_ERROR(cudaFree(d_bwsup_hid));
    CUDA_CHECK_ERROR(cudaFree(d_supinf_hid));
    CUDA_CHECK_ERROR(cudaFree(d_sup_hid));
    CUDA_CHECK_ERROR(cudaFree(d_act_hid));
    CUDA_CHECK_ERROR(cudaFree(d_pnoise_hid));
    CUDA_CHECK_ERROR(cudaFree(d_rndPoisson_hid));
    CUDA_CHECK_ERROR(cudaFree(d_ada_hid));
    CUDA_CHECK_ERROR(cudaFree(d_sada_hid));
    CUDA_CHECK_ERROR(cudaFree(d_axoact_ih));
    CUDA_CHECK_ERROR(cudaFree(d_denact_ih));
    CUDA_CHECK_ERROR(cudaFree(d_Hihjhi_ih));
    CUDA_CHECK_ERROR(cudaFree(d_trgpopact_ih));
    CUDA_CHECK_ERROR(cudaFree(d_bwsup_ih));
    CUDA_CHECK_ERROR(cudaFree(d_bwsupinf_ih));
    CUDA_CHECK_ERROR(cudaFree(d_Zj_ih));
    CUDA_CHECK_ERROR(cudaFree(d_Zi_ih));
    CUDA_CHECK_ERROR(cudaFree(d_Pj_ih));
    CUDA_CHECK_ERROR(cudaFree(d_Pi_ih));
    CUDA_CHECK_ERROR(cudaFree(d_Pji_ih));
    CUDA_CHECK_ERROR(cudaFree(d_Bj_ih));
    CUDA_CHECK_ERROR(cudaFree(d_Wji_ih));
    CUDA_CHECK_ERROR(cudaFree(d_hmax));
    CUDA_CHECK_ERROR(cudaFree(d_hsum));    

    CUDA_CHECK_ERROR(cudaFree(d_lgi_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_bwsup_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_supinf_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_sup_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_act_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_pnoise_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_rndPoisson_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_ada_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_sada_hid_opt));
    CUDA_CHECK_ERROR(cudaFree(d_axoact_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_denact_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_Hihjhi_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_trgpopact_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_bwsup_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_bwsupinf_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_Zj_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_Zi_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_Pj_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_Pi_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_Pji_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_Bj_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_Wji_ih_opt));
    CUDA_CHECK_ERROR(cudaFree(d_hmax_opt));
    CUDA_CHECK_ERROR(cudaFree(d_hsum_opt));

    return 0;
}