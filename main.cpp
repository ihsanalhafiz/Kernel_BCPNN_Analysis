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

const int NumberPop = 3;
const float eps_hls = 1e-8;
const float eps_neg_hls = -1e-8;
const float EPS_GLB = 1e-7;

// Layer Population Input
const int H_in = 784;
const int M_in = 2;
const int N_in = H_in * M_in;

// Layer Population Hidden
const int H_hid = 32;
const int M_hid = 128;
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

    float* Zj_ih = new float[N_hid];
    float* Zi_ih = new float[H_hid * denNi_ih];
    float* Pj_ih = new float[N_hid];
    float* Pi_ih = new float[H_hid * denNi_ih];
    float* Pji_ih = new float[N_hid * denNi_ih];
    float* Bj_ih = new float[N_hid];
    float* Wji_ih = new float[N_hid * denNi_ih];

    // Seed for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

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
        Hihjhi_ih[i] = static_cast<int>(dis(gen) * std::numeric_limits<int>::max());
    }


    return 0;
}