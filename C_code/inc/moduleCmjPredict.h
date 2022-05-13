#include <stdint.h>

#define ACC_IN_LENGTH 1000
#define ACC_CUT_LENGTH 400
#define ACC_FREQ 100
#define EPSILON 0.000001

#define CMJ_MODEL_LEN 7
#define SLJ_MODEL_LEN 2
#define SPRINT_50_MODEL_LEN 3

struct ModuleCmjPredict {
    /* input acc */
    float acc_x[ACC_IN_LENGTH];
    float sig_cut[ACC_CUT_LENGTH];

    /* linear regression parameter for cmj prediction */
    float cmj_model[7];
    float slj_model[2];
    float sprint_50_model[3];
    
    /* features for cmj prediction */
    float feat_8;
    float feat_9;
    uint16_t feat_16;
    uint16_t feat_33;
    uint16_t feat_41;
    
    /* prediction value */
    float cmj_pred;
    float slj_pred;
    float sprint_50_pred;
};

void getCmjPredictVersion(int *version);
void initCmjPredict(struct ModuleCmjPredict *cmjPredict);
void triggerOn(struct ModuleCmjPredict *cmjPredict, uint32_t dt, float *input_acc_x);
uint8_t triggerOff(struct ModuleCmjPredict *cmjPredict, float height, float history_slj, float history_sprint_50);
