#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "inc/moduleCmjPredict.h"
#include "inc/moduleSgnFilters.h"


int main(void) {
    FILE *acc_in, *acc_filt_out;
    mBWpara BandPass;
    float BP_cutoff[2] = {2.0f / (100.0f / 2.0f), 0};

    initSngBWfilter(&BandPass, 1, 2, BP_cutoff);

    acc_in = fopen("sub8_test7_IBSH.txt", "r");

    float acc_x[ACC_IN_LENGTH];
    float acc_tmp;
    int time = 0;
    
    while (!feof(acc_in)) {
        fscanf(acc_in, "%f", &acc_tmp);
        acc_x[time] = acc_tmp;
        // printf("%f\n", acc_tmp);
        time ++;
    }
    fclose (acc_in);

    filterBW(&BandPass, acc_x, ACC_IN_LENGTH);

    acc_filt_out = fopen("sub8_test7_IBSH_C.txt", "w");
    if (acc_filt_out == NULL) {
        printf("create out file error!");
        exit(1);
    }
    
    for (int i=0; i<ACC_IN_LENGTH; i++){
        fprintf(acc_filt_out, "%f\n", acc_x[i]);
    }
    printf("write end.");
    fclose (acc_filt_out);


    return 0;
}