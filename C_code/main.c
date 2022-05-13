#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "inc/moduleCmjPredict.h"
#include "inc/mathFunction.h"

int main(void) {
    FILE *acc_in;
    float acc_x[ACC_IN_LENGTH];
    float acc_tmp;
    int idx = 0;
    int time = 0;
    uint8_t error_code = 0;
    uint8_t dt = 1;

    float height = 168.0;
    uint8_t gender = 1;
    uint8_t grade = 8;
    float history_slj = -1.0f;
    float history_sprint_50 = -1.0f;

    struct ModuleCmjPredict cmjPredict;
    initCmjPredict(&cmjPredict);
    
    /* I/O get acc test data */
    acc_in = fopen("sub7_test5_YCSH.txt", "r");
    while (!feof(acc_in)) {
        fscanf(acc_in, "%f", &acc_tmp);
        acc_x[idx] = acc_tmp;
        idx ++;

        if (idx % 100 == 0) {
            triggerOn(&cmjPredict, 1, acc_x);
            idx = 0;
            time ++;
        }
    }
    fclose (acc_in);
    triggerOff(&cmjPredict, height, history_slj, history_sprint_50);
    
    // printf("feat_8: %f\n", cmjPredict.feat_8);
    // printf("feat_9: %f\n", cmjPredict.feat_9);
    // printf("feat_16: %d\n", cmjPredict.feat_16);
    // printf("feat_33: %d\n", cmjPredict.feat_33);
    // printf("feat_41: %d\n", cmjPredict.feat_41);
    printf("cmj_pred: %f\n", cmjPredict.cmj_pred);
    printf("slj_pred: %f\n", cmjPredict.slj_pred);
    printf("sprint_50_pred: %f\n", cmjPredict.sprint_50_pred);
    printf("error_code: %d\n", error_code);

    return 0;
}