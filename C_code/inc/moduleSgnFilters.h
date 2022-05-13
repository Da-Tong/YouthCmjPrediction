#ifndef SGN_FILTERS_H
#define SGN_FILTERS_H
#include <stdint.h>

// Constants
#define PI 3.141592653589793238462643383279502884197169399375105

// MARCO
#define DIV(x, y) (y==0 ? (0) : (x/y))
#define IS_HR_VALID(hr) ((hr) >= 40 && (hr) <= 240)
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(x,y) ( x>y ? y : x )

#define MAX_FILTER_LENGTH 8
typedef struct BWpara {
    int filter_length;
    int initialed;
    // Butterworth Coefficients
    float a[MAX_FILTER_LENGTH + 1];
    float b[MAX_FILTER_LENGTH + 1];
    float input_prev[MAX_FILTER_LENGTH];
    float output_prev[MAX_FILTER_LENGTH];
} mBWpara;

struct ModuleSgnFilters {
    mBWpara BWparaPPG;
    mBWpara BWparaAccX;
    mBWpara BWparaAccY;
    mBWpara BWparaAccZ;
};

void getSgnFilterVersion(int *version);

void sngFlushBW(mBWpara *SgnBWpara);

void initSngBWfilter(mBWpara *SgnBWpara, int type, int order, float * fc);

void filterBW(mBWpara *SgnBWpara, float *input_signal, int sgn_len);

void resampleSGN(float *signal, int fs_in, int fs_out, int signal_size, float *output_signal);

void zScoreNorm(float *signal, int signal_size, float *output_signal);

void DcRemoveMin(float *signal, int signal_size, float *output_signal);
#endif
