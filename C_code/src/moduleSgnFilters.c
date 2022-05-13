#include <string.h>
#include <math.h>
#include "../inc/moduleSgnFilters.h"

void getSgnFilterVersion(int *version) {
    version[0] = 2021009; /* release date */
    version[1] = 1; /* version */
    version[2] = 2; /* subversion */
    version[3] = 0; /* debug build */
}

static void initLPHP(mBWpara *SgnBWpara, int type, int order, float fc) {
//    float temp, e_pole_r, e_pole_i, e_step_r, e_step_i, r_r[2], r_i[2], a_real[2],
//	a_imag[2], s = 1.0f;

    float temp;
    float e_pole_r, e_pole_i;
    float e_step_r, e_step_i;
    float r_r[MAX_FILTER_LENGTH], r_i[MAX_FILTER_LENGTH];
    float a_real[MAX_FILTER_LENGTH];
    float a_imag[MAX_FILTER_LENGTH];
    float s = 1.0;
    
	SgnBWpara->initialed = 0;

    e_step_r = cosf(PI * fc);
    e_step_i = sinf(PI * fc);

    for (int i = 0; i < order; ++i) {
         temp = PI * (float) (2 * i + 1) / (float) (2 * order);

         e_pole_r = cosf(temp);
         e_pole_i = sinf(temp);
         if(i < order / 2) s *= 1.0f + e_step_i * e_pole_i;
        
         // ======= Place Poles =======
         temp = 1.0f + e_step_i * e_pole_i;
         r_r[i] = -e_step_r / temp;
         r_i[i] = -e_step_i * e_pole_r / temp;
    }

    a_real[0] = r_r[0];
    a_imag[0] = r_i[0];

    for (int i = 1; i < order; i++) {
         a_real[i] = 0.0f;
         a_imag[i] = 0.0f;
         for (int j = i; j > 0; j--) {
              a_real[j] += a_real[j - 1] * r_r[i] - a_imag[j - 1] * r_i[i];
              a_imag[j] += a_real[j - 1] * r_i[i] + a_imag[j - 1] * r_r[i];
         }
         a_real[0] += r_r[i];
         a_imag[0] += r_i[i];
    }

    SgnBWpara->b[0] = 1.0f;
    for (int i = 1; i <= order; i++) {
        SgnBWpara->b[i] = a_real[i - 1];
    }
    
    // === Scale Factor Calculation ===
    e_step_r = cosf(PI * fc / 2.0f);
    e_step_i = sinf(PI * fc / 2.0f);
    if (order % 2) s *= e_step_r + e_step_i;

    if (type == 0) {
        SgnBWpara->a[0] = powf(e_step_i, (float)order) / s;
    } else {
        SgnBWpara->a[0] = powf(e_step_r, (float)order) / s;
    }
    SgnBWpara->a[1] = SgnBWpara->a[0];

    for (int i = 1; i < order; i++) {
         SgnBWpara->a[i + 1] = 0.0f;
         for (int j = i + 1; j > 0; j--) {
              SgnBWpara->a[j] += SgnBWpara->a[j - 1];
        }
    }

    if (type) {
        for(int i = 1; i < order + 1; i += 2) SgnBWpara->a[i] *= 0-1;
    }
}

static void initBP(mBWpara *SgnBWpara, int order, float *fc) {
//    float cp, temp, e_step_r, e_step_i, e_2step_r, e_2step_i, e_pole_r, e_pole_i;
//    float r1_r[2], r1_i[2];  // coefficient for z^-1
//    float r2_r[2], r2_i[2];  // coefficient for z^-2
//    float a_real [2 * 2], a_imag [2 * 2], s_r, s_i, s_r_buf, ct;


    float cp;
    float temp;
    float e_step_r, e_step_i;
    float e_2step_r, e_2step_i;
    float e_pole_r, e_pole_i;
    float r1_r[MAX_FILTER_LENGTH / 2], r1_i[MAX_FILTER_LENGTH / 2];  // coefficient for z^-1
    float r2_r[MAX_FILTER_LENGTH / 2], r2_i[MAX_FILTER_LENGTH / 2];  // coefficient for z^-2
    float a_real [MAX_FILTER_LENGTH];
    float a_imag [MAX_FILTER_LENGTH];

    float s_r, s_i, s_r_buf;
    float ct;
    
    SgnBWpara->initialed = 0;
    cp = cosf(PI * (fc[0] + fc[1]) / 2.0f);
    e_step_r = cosf(PI * (fc[1] - fc[0]) / 2.0f);
    e_step_i = sinf(PI * (fc[1] - fc[0]) / 2.0f);
    e_2step_r = cosf(PI * (fc[1] - fc[0]));
    e_2step_i = sinf(PI * (fc[1] - fc[0]));
    ct = 1.0f / tanf(PI * (fc[1] - fc[0]) / 2.0f);

    s_r = 1.0f;
    s_i = 0.0f;

    for (int i = 0; i < order; ++i) {
         temp = PI * (float) (2 * i + 1) / (float) (2 * order);
         e_pole_r = cosf(temp);
         e_pole_i = sinf(temp);
        
         // Scaling Factor Calculation
         s_r_buf = s_r;
         s_r = s_r * (e_pole_i + ct) + s_i * e_pole_r;
         s_i = (s_r_buf + s_i) * (ct + e_pole_i - e_pole_r) - s_r_buf * (e_pole_i + ct) + s_i * e_pole_r;
        
         temp = 1.0f + e_2step_i * e_pole_i;
         r1_r[i] = e_2step_r / temp;
         r1_i[i] = e_2step_i * e_pole_r / temp;
         r2_r[i] = -2.0f * cp * (e_step_r + e_step_i * e_pole_i) / temp;
         r2_i[i] = -2.0f * cp * e_step_i * e_pole_r / temp;
    }

    a_real[0] = r2_r[0];
    a_imag[0] = r2_i[0];
    a_real[1] = r1_r[0];
    a_imag[1] = r1_i[0];
    
    for (int i = 1; i < order; i++) {
         a_real[2 * i] = a_real[2 * i + 1] = 0.0f;
         a_imag[2 * i] = a_imag[2 * i + 1] = 0.0f;

         for (int j = i; j > 0; j--) {
              a_real[2 * j + 1] += a_real[2 * j - 1] * r1_r[i] + a_real[2 * j] * r2_r[i]
                                - a_imag[2 * j - 1] * r1_i[i] - a_imag[2 * j] * r2_i[i];
              a_imag[2 * j + 1] += a_real[2 * j - 1] * r1_i[i] + a_imag[2 * j] * r2_r[i]
                                + a_imag[2 * j - 1] * r1_r[i] + a_real[2 * j] * r2_i[i];

              a_real[2 * j] += a_real[2 * j - 2] * r1_r[i] + a_real[2 * j - 1] * r2_r[i]
                            - a_imag[2 * j - 2] * r1_i[i] - a_imag[2 * j - 1] * r2_i[i];
              a_imag[2 * j] += a_real[2 * j - 2] * r1_i[i] + a_real[2 * j - 1] * r2_i[i]
                            + a_imag[2 * j - 2] * r1_r[i] + a_imag[2 * j - 1] * r2_r[i];
         }

         a_real[1] += r1_r[i] + a_real[0] * r2_r[i] - a_imag[0] * r2_i[i];
         a_imag[1] += r1_i[i] + a_real[0] * r2_i[i] + a_imag[0] * r2_r[i];
         a_real[0] += r2_r[i];
         a_imag[0] += r2_i[i];
    }

    SgnBWpara->b[0] = 1.0f;
    for (int i = 1; i <= order * 2; i++) {
         SgnBWpara->b[i] = a_real[i - 1];
    }
    
    SgnBWpara->a[0] = 1.0f / s_r;
    SgnBWpara->a[2] = 1.0f / s_r;

    for (int i = 1; i < order; i ++) {
        SgnBWpara->a[2 * i + 2] = SgnBWpara->a[2 * i + 1] = 0.0f;
        for (int j = i + 1; j > 0; j--) {
             SgnBWpara->a[2 * j] += SgnBWpara->a[2 * j - 2];
        }
    }

    for (int i = 2; i < order * 2; i += 4) {
         SgnBWpara->a[i] *= -1.0;
    }

}

void sngFlushBW(mBWpara *SgnBWpara) {
    for (int i = 0; i < MAX_FILTER_LENGTH; i++) {
         SgnBWpara->input_prev[i] = SgnBWpara->output_prev[i] = 0.0f;
    }
}


/********
* type: filter type
* 0 - Lowpass
* 1 - Highpass
* 2 - Bandpass
* order: filter order
* max filter order for LP, HP: 8
* max filter order for BP: 4
* fc: cutoff frequency (unit: freq / (sampling_freq / 2))
* fc[0]: 1st cutoff frequency
* fc[1]: 2nd cutoff frequency (only needed for bandpass filter)
* ******/
void initSngBWfilter(mBWpara *SgnBWpara, int type, int order, float * fc) {
    if (type < 2) {
        // Low Pass Filter & High Pass Filter
        initLPHP(SgnBWpara, type, order, fc[0]);
        SgnBWpara->filter_length = order;
    } else if (type == 2) {
        // Bandpass Filter
        initBP(SgnBWpara, order, fc);
        SgnBWpara->filter_length = order * 2;
    }
    
    sngFlushBW(SgnBWpara);
}

void filterBW(mBWpara *SgnBWpara, float *input_signal, int sgn_len) {  
    float buf = 0.0f;
    
    for(int i = 0; i < sgn_len; i++) {
        buf = SgnBWpara->a[0] * input_signal[i];
        for(int j = 1; j <= SgnBWpara->filter_length; j++) {
            buf += SgnBWpara->input_prev[j - 1] * SgnBWpara->a[j];
            if(i - j >= 0) buf -= input_signal[i - j] * SgnBWpara->b[j];
            else buf -= SgnBWpara->output_prev[j - i - 1] * SgnBWpara->b[j];
        }
        
        for(int j = SgnBWpara->filter_length - 1; j > 0; j--) {
            SgnBWpara->input_prev[j] = SgnBWpara->input_prev[j - 1];
        }
        SgnBWpara->input_prev[0] = input_signal[i];
        input_signal[i] = buf;
    }
    
    if (sgn_len < SgnBWpara->filter_length) {
        // Shift original output buf
        for(int i = SgnBWpara->filter_length - 1; i >= sgn_len; i--) {
            SgnBWpara->output_prev[i] = SgnBWpara->output_prev[i - sgn_len];
        }
        // Record new output
        for(int i = 0; i < sgn_len; i++) {
            SgnBWpara->output_prev[i] = input_signal[sgn_len - i - 1];
        }

    } else {
        for(int i = 0; i < SgnBWpara->filter_length; i++) {
            SgnBWpara->output_prev[i] = input_signal[sgn_len - i - 1];
        }
    }
    SgnBWpara->initialed = 1;
}

void resampleSGN(float *signal, int fs_in, int fs_out, int signal_size, float *output_signal) {
    if(fs_in == fs_out) memcpy(output_signal, signal, sizeof(float)*signal_size);
    else
    {
        int old_index_end = (int)(floor((signal_size / fs_in) * fs_out));
        float interval = 1.0f / fs_out - 1.0f / fs_in; //if positive: high freq -> low freq; if negtive: low->high
        int downsample = 0, last_pt = 0, next_pt = 0;
        float interp_ratio = 0, pt_diff = 0, interp = 0, new_index = 0, old_index = 0;;
        
        if ((1.0f / fs_out) - (1.0f / fs_in) > 0) downsample = 1;

        for (new_index = 0; new_index < old_index_end; new_index++) {
            
            if (interval < 0) interval = interval * (-1);

            if ((int)(new_index * fs_in) % fs_out == 0) {
                output_signal[(int)new_index] = signal[(int)(new_index * fs_in / fs_out)];
            } else {
                if (downsample) {
                    last_pt = (int)((new_index / fs_out - interval) * fs_in);
                    next_pt = last_pt + 1;
                    if (next_pt > signal_size - 1)
                        next_pt = signal_size - 1;
                    interp_ratio = interval * fs_out;
                    pt_diff = signal[next_pt] - signal[last_pt];
                    interp = pt_diff * interp_ratio;
                    output_signal[(int)new_index] = signal[last_pt] + interp;
                    old_index = (float)next_pt;
                } else {
                    next_pt = (int)((new_index / fs_out + interval) * fs_in);
                    last_pt = next_pt - 1;
                    interp_ratio = interval * fs_out;
                    pt_diff = signal[next_pt] - signal[last_pt];
                    interp = pt_diff * interp_ratio;
                    output_signal[(int)new_index] = signal[next_pt] - interp;
                    old_index = (float)next_pt;
                }
            }
        }
    }
    
}

void zScoreNorm(float *signal, int signal_size, float *output_signal) {
    int16_t i;
    float sgn_mean = 0.0f;
    float sgn_std = 0.0f;
    for(i=0; i<signal_size; i++) {
        sgn_mean += signal[i];
    }
    sgn_mean = DIV(sgn_mean,(float)signal_size);
    for(i=0; i<signal_size; i++) {
        sgn_std += (signal[i]-sgn_mean)*(signal[i]-sgn_mean);
    }
    sgn_std = DIV(sgn_std,(signal_size-1));
    sgn_std = sqrtf(sgn_std);
    for(i=0; i<signal_size; i++) {
        output_signal[i] = DIV((signal[i]-sgn_mean),sgn_std);
    }
}

void DcRemoveMin(float *signal, int signal_size, float *output_signal) {
    int16_t i;
    float sgn_min = signal[0];
    for(i=1; i<signal_size; i++) {
        if(signal[i]<sgn_min) sgn_min = signal[i];
    }
    for(i=0; i<signal_size; i++) {
        output_signal[i] = signal[i]-sgn_min;
    }
}

void diff(float *input_array, float* output_array, int array_len){
    for (int i = 0; i < array_len; i++){
        output_array[i] = input_array[i+1]-input_array[i];
    }
    if(array_len > 0) output_array[array_len-1] = 0;
}

void gradient(float *input_array, float* output_array, int array_len){
    if(array_len >= 2){
        float left_val, diff;
        for (int i = 0; i < array_len; i++){

            if(i == 0) diff = input_array[i+1]-input_array[i];
            else if(i == array_len -1) diff = input_array[i]-left_val;
            else diff = ((input_array[i]-left_val)+(input_array[i+1]-input_array[i]))/2;

            left_val = input_array[i];
            output_array[i] = diff;
        }
    }else{
        if(array_len > 0) output_array[0] = 0;
    }
}
