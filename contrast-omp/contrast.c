#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "hist-equ.h"

void run_cpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);


int main(int argc, char *argv[]){
    double tstart = omp_get_wtime();

    fprintf(stderr, "OpenMP: max threads = %d, procs = %d\n",
            omp_get_max_threads(), omp_get_num_procs());

    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;
    
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm("images/in.pgm");
    run_cpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g);
    
    printf("Running contrast enhancement for color images.\n");
    img_ibuf_c = read_ppm("images/in.ppm");
    run_cpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    double tfinish = omp_get_wtime();
    printf("Total application time: %f (ms)\n", (tfinish - tstart)*1000.0);

    return 0;
}

void run_cpu_color_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_hsl, img_obuf_yuv;

    
    printf("Starting CPU processing...\n");
    
    double tstart = omp_get_wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    double tfinish = omp_get_wtime();
    printf("HSL processing time: %f (ms)\n", (tfinish - tstart)*1000.0);
    
    write_ppm(img_obuf_hsl, "images/out_hsl_omp.ppm");

    double tstart2 = omp_get_wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    double tfinish2 = omp_get_wtime();
    printf("YUV processing time: %f (ms)\n", (tfinish2 - tstart2)*1000.0);
    
    write_ppm(img_obuf_yuv, "images/out_yuv_omp.ppm");
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}




void run_cpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;
    

    printf("Starting CPU processing...\n");
    
    double tstart = omp_get_wtime();
    img_obuf = contrast_enhancement_g(img_in);
    double tfinish = omp_get_wtime();
    printf("Processing time: %f (ms)\n", (tfinish - tstart)*1000.0);
    
    write_pgm(img_obuf, "images/out_omp.pgm");
    free_pgm(img_obuf);
}



PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    unsigned char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    if (fscanf(in_file, "%s", sbuf) != 1) exit(1);

    if (fscanf(in_file, "%d",&result.w) != 1) exit(1);
    if (fscanf(in_file, "%d",&result.h) != 1) exit(1);
    if (fscanf(in_file, "%d\n",&v_max) != 1) exit(1);
    printf("Image size: %d x %d\n", result.w, result.h);
    
    int n = result.w * result.h;
    result.img_r = (unsigned char *)malloc(n * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(n * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(n * sizeof(unsigned char));
    ibuf         = (unsigned char *)malloc(3 * n * sizeof(unsigned char));

    // File read is sequential (E/S); only the distribution of the channels is parallelized
    if (fread(ibuf,sizeof(unsigned char), 3 * n, in_file) != (size_t)(3*n)) exit(1);

    #pragma omp parallel for schedule(static)
    for(i = 0; i < n; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    int n = img.w*img.h;
    
    unsigned char * obuf = (unsigned char *)malloc(3 * n * sizeof(unsigned char));

    #pragma omp parallel for schedule(static)
    for(i = 0; i < n; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*n, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    PGM_IMG result;
    int v_max;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    if (fscanf(in_file, "%s", sbuf) != 1) exit(1); /*Skip the magic number*/
    if (fscanf(in_file, "%d",&result.w) != 1) exit(1);
    if (fscanf(in_file, "%d",&result.h) != 1) exit(1);
    if (fscanf(in_file, "%d\n",&v_max) != 1) exit(1);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    if (fread(result.img,sizeof(unsigned char), result.w*result.h, in_file) != (size_t)(result.w*result.h)) exit(1);
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}