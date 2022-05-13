#include <stdint.h>

float cal_sum(const float *list, uint8_t front, uint8_t end);

float cal_std(float *temp, uint8_t count);

void trap_intergral_1d(float *data, int16_t data_len, int fz);

float linear_function(float *model_a, float model_b, float *feats, int8_t feat_len);

uint8_t is_local_min(float now, float prev_1, float prev_2, float next_1, float next_2);

uint8_t is_local_max(float now, float prev_1, float prev_2, float prev_3, float next_1, float next_2, float next_3);