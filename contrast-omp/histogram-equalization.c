#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "hist-equ.h"

// Every thread accumulates in a private copy of the histogram (reduction over
// array section, OpenMP >= 4.5) and at the end all copies are summed.
// Since it's integer arithmetic, the result is exactly the sequential one and we
// avoid both contention and false sharing on hist_out.

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    #pragma omp parallel for schedule(static) reduction(+:hist_out[:nbr_bin])
    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    unsigned char *lut = (unsigned char *)malloc(sizeof(unsigned char)*nbr_bin);
    int i, cdf, min, d, v;

    /* Construct the LUT by calculating the CDF */
    // There are only nbr_bin (256) elements with dependency (accumulative sum):
    // so it doesn't pay off to parallelize it, it is done in series (same formula as the
    // sequential version to obtain identical results).
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        v = (int)(((float)cdf - min)*255/d + 0.5);
        if(v < 0)   v = 0;
        if(v > 255) v = 255;      // clamp is only applied once here 
        lut[i] = (unsigned char)v;
    }

    // Application of the LUT -> completely parallel loop (map)
    #pragma omp parallel for schedule(static)
    for(i = 0; i < img_size; i ++){
        img_out[i] = lut[img_in[i]];
    }

    free(lut);
}