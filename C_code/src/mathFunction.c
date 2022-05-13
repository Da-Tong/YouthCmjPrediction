#include "../inc/mathFunction.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

float cal_sum(const float *list, uint8_t front, uint8_t end) {
    float sum=0.0f;

	for (uint8_t i=front;i<end;i++) {
         sum+=list[i];
	}
	return sum;
}

float cal_std(float *temp, uint8_t count) {
    float s_avg = 0, avg;

    if (count == 1) {
        return 0.0f;
	} 
    else {
        avg = cal_sum(temp, 0, count) / count;

	    for (uint8_t i=0;i<count;i++){
             s_avg += powf(temp[i] - avg, 2);
        }
        float std = sqrtf(s_avg / (count-1));
		return std;
    }
}

void trap_intergral_1d(float *data, int16_t data_len, int fz) {
    float tmp=0;
    float h = 1.0f / fz;
    float f1, f2;
    
    f1 = data[0];
    data[0] = 0.0f;
    for (int i=1; i<data_len; i++) {
        f2 = data[i];
        tmp += (f1 + f2) * h / 2;
        f1 = data[i];
        data[i] = tmp;
    }
}

float linear_function(float *model_a, float model_b, float *feats, int8_t feat_len) {
    float pred_value=0;

    if (sizeof(model_a) != sizeof(feats)) {
        printf("input length mismatch!\n");
        exit(0);
    }
    else {
        for (int8_t i=0; i<feat_len; i++) {
            pred_value += model_a[i] * feats[i];
        }
        pred_value += model_b;
    }
    return pred_value;
}

uint8_t is_local_min(float now, float prev_1, float prev_2, float next_1, float next_2) {
    if ((now < prev_1) && (now < next_1)) {
        if ((now < prev_2) && (now < next_2)) {
            return 1;
        }
    }
    return 0;
}
    
uint8_t is_local_max(float now, float prev_1, float prev_2, float prev_3, float next_1, float next_2, float next_3) {
    if ((now > prev_1) && (now > next_1)) {
        if ((now > prev_2) && (now > next_2)) {
            if ((now > prev_3) && (now > next_3)) {
                return 1;
            }
        }
    }
    return 0;
}
    