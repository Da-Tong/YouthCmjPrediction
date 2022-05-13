#include "../inc/moduleCmjPredict.h"
#include "../inc/moduleSgnFilters.h"
#include "../inc/mathFunction.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

void getCmjPredictVersion(int *version) {
    version[0] = 20211217;
    version[1] = 1;
    version[2] = 0;
    version[3] = 0;
}

static void initInputAcc(struct ModuleCmjPredict *cmjPredict) {
    memset(cmjPredict->acc_x, 0, sizeof(cmjPredict->acc_x));
    memset(cmjPredict->sig_cut, 0, sizeof(cmjPredict->sig_cut));
}

static void initRegressor(struct ModuleCmjPredict *cmjPredict) {
    cmjPredict->cmj_model[0] = 0.209622;
    cmjPredict->cmj_model[1] = -1.036122;
    cmjPredict->cmj_model[2] = -0.409013;
    cmjPredict->cmj_model[3] = 0.123294;
    cmjPredict->cmj_model[4] = -0.0638;
    cmjPredict->cmj_model[5] = 0.008116;
    cmjPredict->cmj_model[6] = -17.585052;

    cmjPredict->slj_model[0] = 3.385828;
    cmjPredict->slj_model[1] = 87.38362;

    cmjPredict->sprint_50_model[0] = -0.035864;
    cmjPredict->sprint_50_model[1] = -0.019978;
    cmjPredict->sprint_50_model[2] = 12.928137;
}

static void initFeatures(struct ModuleCmjPredict *cmjPredict) {
    cmjPredict->feat_8 = -1.0f;
    cmjPredict->feat_9 = -1.0f;
    cmjPredict->feat_16 = 0;
    cmjPredict->feat_33 = 0;
    cmjPredict->feat_41 = 0;
}

static void initOutput(struct ModuleCmjPredict *cmjPredict) {
    cmjPredict->cmj_pred = -1.0f;
    cmjPredict->slj_pred = -1.0f;
    cmjPredict->sprint_50_pred = -1.0f;
}

static void updateAccX(struct ModuleCmjPredict *cmjPredict, float *intput_acc_x) {
    for (int i=0; i<ACC_IN_LENGTH-ACC_FREQ; i++) {
        cmjPredict->acc_x[i] = cmjPredict->acc_x[i+ACC_FREQ];
    }
    for (int i=0; i<ACC_FREQ; i++) {
        cmjPredict->acc_x[i + ACC_IN_LENGTH - ACC_FREQ] = intput_acc_x[i];
    }
    
}

static uint8_t cmj_check(struct ModuleCmjPredict *cmjPredict) {
    uint8_t error_code = 0;
    int min_idx = 0;
    float min_value = cmjPredict->acc_x[0];

    for (int i=1; i<ACC_IN_LENGTH; i++) {
        if (cmjPredict->acc_x[i] < min_value) {
            min_idx = i;
            min_value = cmjPredict->acc_x[i];
        }
    }

    if ((min_idx < 2 * ACC_FREQ) && (min_idx > ACC_IN_LENGTH - 2 * ACC_FREQ)) {
        return 1;
    }

    for (int i=0; i<ACC_CUT_LENGTH; i++) {
        cmjPredict->sig_cut[i] = cmjPredict->acc_x[i + min_idx - 2 * ACC_FREQ];
    }
    
    if (cal_std(cmjPredict->sig_cut, 0.5 * ACC_FREQ) > 0.98) {
        return 1;
    }

    if ((cal_sum(cmjPredict->sig_cut, 0, 0.5 * ACC_FREQ) / (0.5 * ACC_FREQ)) > 4.9) {
        return 1;
    }

    return 0;
}

static uint8_t get_vel_feature(struct ModuleCmjPredict *cmjPredict) {
    uint8_t error_code = 0;
    mBWpara BandPass;
    int16_t x_vel_idxs[6];
    memset(x_vel_idxs, -1, sizeof(x_vel_idxs));
    
    /* filt acc_x and intergral to velocity */
    float BP_cutoff[2] = {2.0f / (100.0f / 2.0f), 0};
    initSngBWfilter(&BandPass, 1, 2, BP_cutoff);
    filterBW(&BandPass, cmjPredict->sig_cut, ACC_CUT_LENGTH);
    trap_intergral_1d(cmjPredict->sig_cut, ACC_CUT_LENGTH, ACC_FREQ);
    
    /* find index of feature */
    for (uint16_t i=110; i<ACC_CUT_LENGTH-6; i++) {
        if ((x_vel_idxs[0] < 0) && (cmjPredict->sig_cut[i] < -0.4)) {
            if  (is_local_min(cmjPredict->sig_cut[i], cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2],
                              cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2])) {
                x_vel_idxs[0] = i;
                continue;
            }
        }

        if ((x_vel_idxs[0] > 0) && (x_vel_idxs[1] < 0) && (cmjPredict->sig_cut[i] > 0.5)) {
            if (is_local_max(cmjPredict->sig_cut[i],
                             cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2], cmjPredict->sig_cut[i-3],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2], cmjPredict->sig_cut[i+3])) {
                x_vel_idxs[1] = i;
                continue;
            }
        }

        if ((x_vel_idxs[1] > 0) && (x_vel_idxs[2] < 0) && (cmjPredict->sig_cut[i] < -1)) {
            if (is_local_min(cmjPredict->sig_cut[i], cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2])) {
                x_vel_idxs[2] = i;
                continue;
            }
        }

        if ((x_vel_idxs[2] > 0) && (x_vel_idxs[3] < 0) && (cmjPredict->sig_cut[i] > 1.5)) {
            if (is_local_max(cmjPredict->sig_cut[i],
                             cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2], cmjPredict->sig_cut[i-6],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2], cmjPredict->sig_cut[i+6])) {
                x_vel_idxs[3] = i;
                continue;
            }
        }

        if ((x_vel_idxs[3] > 0) && (x_vel_idxs[4] < 0) && (cmjPredict->sig_cut[i] < 0.25)) {
            if (is_local_min(cmjPredict->sig_cut[i], cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2])) {
                x_vel_idxs[4] = i;
                continue;
            }
        }

        if ((x_vel_idxs[4] > 0) && (x_vel_idxs[5] < 0) && (cmjPredict->sig_cut[i] > 0.25)) {
            if (is_local_max(cmjPredict->sig_cut[i],
                             cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2], cmjPredict->sig_cut[i-3],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2], cmjPredict->sig_cut[i+3])) {
                x_vel_idxs[5] = i;
                break;
            }
        }
    }

    /* check is feature valid */
    if (x_vel_idxs[5] < 0) {
        error_code = 2;
    }

    /* get velocity features */
    cmjPredict->feat_8 = cmjPredict->sig_cut[x_vel_idxs[2]] - cmjPredict->sig_cut[x_vel_idxs[1]];
    cmjPredict->feat_9 = cmjPredict->sig_cut[x_vel_idxs[3]] - cmjPredict->sig_cut[x_vel_idxs[2]];
    cmjPredict->feat_16 = x_vel_idxs[5] - x_vel_idxs[4];
    return error_code;
}

static uint8_t get_disp_feature(struct ModuleCmjPredict *cmjPredict) {
    uint8_t error_code = 0;
    int16_t x_disp_idxs[6];
    memset(x_disp_idxs, -1, sizeof(x_disp_idxs));
    
    /* intergral velocity to displacement */
    trap_intergral_1d(cmjPredict->sig_cut, ACC_CUT_LENGTH, ACC_FREQ);

    /* find index of feature */
    for (uint16_t i=110; i<ACC_CUT_LENGTH-6; i++) {
        if ((x_disp_idxs[0] < 0) && (cmjPredict->sig_cut[i] < -0.1)) {
            if  (is_local_min(cmjPredict->sig_cut[i], cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2],
                              cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2])) {
                x_disp_idxs[0] = i;
                continue;
            }
        }

        if ((x_disp_idxs[0] > 0) && (x_disp_idxs[1] < 0) && (cmjPredict->sig_cut[i] > cmjPredict->sig_cut[x_disp_idxs[0]])) {
            if (is_local_max(cmjPredict->sig_cut[i],
                             cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2], cmjPredict->sig_cut[i-3],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2], cmjPredict->sig_cut[i+3])) {
                x_disp_idxs[1] = i;
                continue;
            }
        }

        if ((x_disp_idxs[1] > 0) && (x_disp_idxs[2] < 0) && (cmjPredict->sig_cut[i] < cmjPredict->sig_cut[x_disp_idxs[0]])) {
            if (is_local_min(cmjPredict->sig_cut[i], cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2])) {
                x_disp_idxs[2] = i;
                continue;
            }
        }

        if ((x_disp_idxs[2] > 0) && (x_disp_idxs[3] < 0) && (cmjPredict->sig_cut[i] > cmjPredict->sig_cut[x_disp_idxs[0]])) {
            if (is_local_max(cmjPredict->sig_cut[i],
                             cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2], cmjPredict->sig_cut[i-6],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2], cmjPredict->sig_cut[i+6])) {
                x_disp_idxs[3] = i;
                continue;
            }
        }

        if ((x_disp_idxs[3] > 0) && (x_disp_idxs[4] < 0) && (cmjPredict->sig_cut[i] > cmjPredict->sig_cut[x_disp_idxs[3]])) {
            if (is_local_max(cmjPredict->sig_cut[i],
                             cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2], cmjPredict->sig_cut[i-6],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2], cmjPredict->sig_cut[i+6])) {
                x_disp_idxs[4] = i;
                continue;
            }
        }

        if ((x_disp_idxs[4] > 0) && (x_disp_idxs[5] < 0) && (cmjPredict->sig_cut[i] < cmjPredict->sig_cut[x_disp_idxs[4]])) {
            if (is_local_min(cmjPredict->sig_cut[i], cmjPredict->sig_cut[i-1], cmjPredict->sig_cut[i-2],
                             cmjPredict->sig_cut[i+1], cmjPredict->sig_cut[i+2])) {
                x_disp_idxs[5] = i;
                continue;
            }
        }
    }

    /* check is feature valid */
    if (x_disp_idxs[5] < 0) {
        error_code = 3;
    }

    /* get velocity features */
    cmjPredict->feat_33 = x_disp_idxs[4] - x_disp_idxs[3];
    cmjPredict->feat_41 = cmjPredict->feat_33 * cmjPredict->feat_33;
    return error_code;
}

static uint8_t acc_feature_extraction(struct ModuleCmjPredict *cmjPredict) {
    /* TO DO */
    uint8_t error_code;

    error_code = get_vel_feature(cmjPredict);
    error_code = get_disp_feature(cmjPredict);
    
    return error_code;
}

static uint8_t preprocess(struct ModuleCmjPredict *cmjPredict) {
    uint8_t error_code = 0;

    error_code = cmj_check(cmjPredict);
    if (error_code == 0) {
        error_code = acc_feature_extraction(cmjPredict);
    }
    else {
        return error_code;
    }
    return error_code;
}

static uint8_t predict_cmj(struct ModuleCmjPredict *cmjPredict, float height) {
    uint8_t error_code = 0;
    float cmj_model_a[CMJ_MODEL_LEN-1];
    float cmj_model_b = cmjPredict->cmj_model[CMJ_MODEL_LEN-1];
    float feats[CMJ_MODEL_LEN-1] = {height, cmjPredict->feat_8, cmjPredict->feat_9, cmjPredict->feat_16, cmjPredict->feat_33, cmjPredict->feat_41};
    
    for (uint8_t i=0; i<CMJ_MODEL_LEN-1; i++) {
        cmj_model_a[i] = cmjPredict->cmj_model[i];
    }

    cmjPredict->cmj_pred = linear_function(cmj_model_a, cmj_model_b, feats, CMJ_MODEL_LEN-1);
    if ((cmjPredict->cmj_pred < 10.0f) || (cmjPredict->cmj_pred > 70.0f)) {
        error_code = 4;
    }
    return error_code;
}

static uint8_t predict_slj(struct ModuleCmjPredict *cmjPredict) {
    uint8_t error_code = 0;
    float slj_model_a[SLJ_MODEL_LEN-1];
    float slj_model_b = cmjPredict->slj_model[SLJ_MODEL_LEN-1];
    float feats[SLJ_MODEL_LEN-1] = {cmjPredict->cmj_pred};
    
    for (uint8_t i=0; i<SLJ_MODEL_LEN-1; i++) {
        slj_model_a[i] = cmjPredict->slj_model[i];
    }

    cmjPredict->slj_pred = linear_function(slj_model_a, slj_model_b, feats, SLJ_MODEL_LEN-1);
    if ((cmjPredict->slj_pred < 50.0f) || (cmjPredict->slj_pred > 330.0f)) {
        error_code = 5;
    }
    return error_code;
}

static uint8_t predict_sprint_50(struct ModuleCmjPredict *cmjPredict) {
    uint8_t error_code = 0;
    float sprint_50_model_a[SPRINT_50_MODEL_LEN-1];
    float sprint_50_model_b = cmjPredict->sprint_50_model[SPRINT_50_MODEL_LEN-1];
    float feats[SPRINT_50_MODEL_LEN-1] = {cmjPredict->cmj_pred, cmjPredict->slj_pred};
    
    for (uint8_t i=0; i<SPRINT_50_MODEL_LEN-1; i++) {
        sprint_50_model_a[i] = cmjPredict->sprint_50_model[i];
    }

    cmjPredict->sprint_50_pred = linear_function(sprint_50_model_a, sprint_50_model_b, feats, SPRINT_50_MODEL_LEN-1);
    if ((cmjPredict->sprint_50_pred < 5.0f) || (cmjPredict->sprint_50_pred > 15.0f)) {
        error_code = 6;
    }
    return error_code;
}

static uint8_t prediction(struct ModuleCmjPredict *cmjPredict, float height) {
    uint8_t error_code = 0;
    
    error_code = predict_cmj(cmjPredict, height);

    if (error_code == 0) {
        error_code = predict_slj(cmjPredict);
    }
    else{return error_code;}

    if (error_code == 0) {
        error_code = predict_sprint_50(cmjPredict);
    }
    else{return error_code;}
    return error_code;
}

static void update_slj(struct ModuleCmjPredict *cmjPredict, float history_slj) {    
    if (history_slj > 0.0f) {
        if (cmjPredict->slj_pred > history_slj) {
            cmjPredict->slj_pred = 0.7f * cmjPredict->slj_pred + 0.3f * history_slj;
        }
        else {
            cmjPredict->slj_pred = 0.3f * cmjPredict->slj_pred + 0.7f * history_slj;
        }
    }
}

static void update_sprint_50(struct ModuleCmjPredict *cmjPredict, float history_sprint_50) {
    if (history_sprint_50 > 0.0f) {
        if (cmjPredict->sprint_50_pred < history_sprint_50) {
            cmjPredict->sprint_50_pred = 0.7f * cmjPredict->sprint_50_pred + 0.3f * history_sprint_50;
        }
        else {
            cmjPredict->sprint_50_pred = 0.3f * cmjPredict->sprint_50_pred + 0.7f * history_sprint_50;
        }
    }
}

void initCmjPredict(struct ModuleCmjPredict *cmjPredict) {
    initInputAcc(cmjPredict);
    initRegressor(cmjPredict);
    initFeatures(cmjPredict);
    initOutput(cmjPredict);
}

void triggerOn(struct ModuleCmjPredict *cmjPredict, uint32_t dt, float *input_acc_x) {
    if (dt == 1) {
        updateAccX(cmjPredict, input_acc_x);
    }
    else {initInputAcc(cmjPredict);}
}

uint8_t triggerOff(struct ModuleCmjPredict *cmjPredict, float height, float history_slj, float history_sprint_50) {
    uint8_t error_code = 0;
    if ((cmjPredict->acc_x[0] - 0.0f > EPSILON) || (0.0f - cmjPredict->acc_x[0] < EPSILON)) {
        error_code = preprocess(cmjPredict);

        if (error_code == 0) {
        error_code = prediction(cmjPredict, height);
        }
        else {return error_code;}

        if (error_code == 0) {
            update_slj(cmjPredict, history_slj);
            update_sprint_50(cmjPredict, history_sprint_50);
        }
    }
    return error_code;
}