#include "canny.h"
#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <stdint.h>	/* for uint64 definition */
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>


#define BILLION 1000000000L
unsigned char filt[N][M], gradient[N][M],grad2[N][M],edgeDir[N][M];
unsigned char gaussianMask[5][5];
signed char GxMask[3][3],GyMask[3][3];
int step_i = 0;
void* image_detection(void* arg){
struct timespec start, end; //timers
uint64_t diff;

int i,j;
	unsigned int    row, col;		
	int rowOffset;					
	int colOffset;					
	int Gx;						
	int Gy;							
	float thisAngle;				
	int newAngle;								
	int newPixel;				
			
        unsigned char temp;					

int core = step_i++;

/* Declare Gaussian mask */
gaussianMask[0][0] = 2;

gaussianMask[0][1] = 4;
gaussianMask[0][2] = 5;
gaussianMask[0][3] = 4;
gaussianMask[0][4] = 2;

gaussianMask[1][0] = 4;
gaussianMask[1][1] = 9;
gaussianMask[1][2] = 12;
gaussianMask[1][3] = 9;
gaussianMask[1][4] = 4;	

gaussianMask[2][0] = 5;
gaussianMask[2][1] = 12;
gaussianMask[2][2] = 15;
gaussianMask[2][3] = 12;
gaussianMask[2][4] = 5;	

gaussianMask[3][0] = 4;
gaussianMask[3][1] = 9;
gaussianMask[3][2] = 12;
gaussianMask[3][3] = 9;
gaussianMask[3][4] = 4;	

gaussianMask[4][0] = 2;
gaussianMask[4][1] = 4;
gaussianMask[4][2] = 5;
gaussianMask[4][3] = 4;
gaussianMask[4][4] = 2;	

/* Declare Sobel masks */
	GxMask[0][0] = -1; GxMask[0][1] = 0; GxMask[0][2] = 1;
	GxMask[1][0] = -2; GxMask[1][1] = 0; GxMask[1][2] = 2;
	GxMask[2][0] = -1; GxMask[2][1] = 0; GxMask[2][2] = 1;
	
	GyMask[0][0] = -1; GyMask[0][1] = -2; GyMask[0][2] = -1;
	GyMask[1][0] =  0; GyMask[1][1] =  0; GyMask[1][2] =  0;
	GyMask[2][0] = 1; GyMask[2][1] = 2; GyMask[2][2] = 1;









/*---------------------- Gaussian Blur ---------------------------------*/
 int xx=0;
 int xy=0;
if (core==0) {xx=(core*256)+2;}
else {xx=(core*256);}
if (core==3) {xy=((core+1)*256)-2;}
else {xy=((core+1)*256);}


__m256i mask1 = _mm256_set_epi8(0,0,gaussianMask[0][4],gaussianMask[0][3],gaussianMask[0][2],gaussianMask[0][1],0,gaussianMask[0][4],gaussianMask[0][3],gaussianMask[0][2],gaussianMask[0][1],gaussianMask[0][0],gaussianMask[0][4],gaussianMask[0][3],gaussianMask[0][2],gaussianMask[0][1],0,gaussianMask[0][4],gaussianMask[0][3],gaussianMask[0][2],gaussianMask[0][1],gaussianMask[0][0],gaussianMask[0][4],gaussianMask[0][3],gaussianMask[0][2],gaussianMask[0][1],0,gaussianMask[0][4],gaussianMask[0][3],gaussianMask[0][2],gaussianMask[0][1],gaussianMask[0][0]);
__m256i mask2 = _mm256_set_epi8(0,0,gaussianMask[1][4],gaussianMask[1][3],gaussianMask[1][2],gaussianMask[1][1],0,gaussianMask[1][4],gaussianMask[1][3],gaussianMask[1][2],gaussianMask[1][1],gaussianMask[1][0],gaussianMask[1][4],gaussianMask[1][3],gaussianMask[1][2],gaussianMask[1][1],0,gaussianMask[1][4],gaussianMask[1][3],gaussianMask[1][2],gaussianMask[1][1],gaussianMask[1][0],gaussianMask[1][4],gaussianMask[1][3],gaussianMask[1][2],gaussianMask[1][1],0,gaussianMask[1][4],gaussianMask[1][3],gaussianMask[1][2],gaussianMask[1][1],gaussianMask[1][0]);
__m256i mask3 = _mm256_set_epi8(0,0,gaussianMask[2][4],gaussianMask[2][3],gaussianMask[2][2],gaussianMask[2][1],0,gaussianMask[2][4],gaussianMask[2][3],gaussianMask[2][2],gaussianMask[2][1],gaussianMask[2][0],gaussianMask[2][4],gaussianMask[2][3],gaussianMask[2][2],gaussianMask[2][1],0,gaussianMask[2][4],gaussianMask[2][3],gaussianMask[2][2],gaussianMask[2][1],gaussianMask[2][0],gaussianMask[2][4],gaussianMask[2][3],gaussianMask[2][2],gaussianMask[2][1],0,gaussianMask[2][4],gaussianMask[2][3],gaussianMask[2][2],gaussianMask[2][1],gaussianMask[2][0]);
__m256i mask4 = _mm256_set_epi8(0,0,gaussianMask[3][4],gaussianMask[3][3],gaussianMask[3][2],gaussianMask[3][1],0,gaussianMask[3][4],gaussianMask[3][3],gaussianMask[3][2],gaussianMask[3][1],gaussianMask[3][0],gaussianMask[3][4],gaussianMask[3][3],gaussianMask[3][2],gaussianMask[3][1],0,gaussianMask[3][4],gaussianMask[3][3],gaussianMask[3][2],gaussianMask[3][1],gaussianMask[3][0],gaussianMask[3][4],gaussianMask[3][3],gaussianMask[3][2],gaussianMask[3][1],0,gaussianMask[3][4],gaussianMask[3][3],gaussianMask[3][2],gaussianMask[3][1],gaussianMask[3][0]);
__m256i mask5 = _mm256_set_epi8(0,0,gaussianMask[4][4],gaussianMask[4][3],gaussianMask[4][2],gaussianMask[4][1],0,gaussianMask[4][4],gaussianMask[4][3],gaussianMask[4][2],gaussianMask[4][1],gaussianMask[4][0],gaussianMask[4][4],gaussianMask[4][3],gaussianMask[4][2],gaussianMask[4][1],0,gaussianMask[4][4],gaussianMask[4][3],gaussianMask[4][2],gaussianMask[4][1],gaussianMask[4][0],gaussianMask[4][4],gaussianMask[4][3],gaussianMask[4][2],gaussianMask[4][1],0,gaussianMask[4][4],gaussianMask[4][3],gaussianMask[4][2],gaussianMask[4][1],gaussianMask[4][0]);
__m256i masl1 = _mm256_set_epi8(0,0,0,0,0,0,gaussianMask[0][0],0,0,0,0,0,0,0,0,0,gaussianMask[0][0],0,0,0,0,0,0,0,0,0,gaussianMask[0][0],0,0,0,0,0);
__m256i masl2 = _mm256_set_epi8(0,0,0,0,0,0,gaussianMask[1][0],0,0,0,0,0,0,0,0,0,gaussianMask[1][0],0,0,0,0,0,0,0,0,0,gaussianMask[1][0],0,0,0,0,0);
__m256i masl3 = _mm256_set_epi8(0,0,0,0,0,0,gaussianMask[2][0],0,0,0,0,0,0,0,0,0,gaussianMask[2][0],0,0,0,0,0,0,0,0,0,gaussianMask[2][0],0,0,0,0,0);
__m256i masl4 = _mm256_set_epi8(0,0,0,0,0,0,gaussianMask[3][0],0,0,0,0,0,0,0,0,0,gaussianMask[3][0],0,0,0,0,0,0,0,0,0,gaussianMask[3][0],0,0,0,0,0);
__m256i masl5 = _mm256_set_epi8(0,0,0,0,0,0,gaussianMask[4][0],0,0,0,0,0,0,0,0,0,gaussianMask[4][0],0,0,0,0,0,0,0,0,0,gaussianMask[4][0],0,0,0,0,0);


__m256i result0 = _mm256_setzero_si256();




clock_gettime(CLOCK_MONOTONIC, &start);
 for(i=0;i<1000;i++){
    for (row = xx; row < xy; row++) {
        for (col = 2; col < M - 30; col+=30) {
        for(j=0; j<5;j++){ 
         __m256i x1=_mm256_loadu_si256((__m256i *) &frame1[row-2][col-2+j]);
         __m256i x2=_mm256_loadu_si256((__m256i *) &frame1[row-1][col-2+j]);   
         __m256i x3=_mm256_loadu_si256((__m256i *) &frame1[row][col-2+j]);
         __m256i x4=_mm256_loadu_si256((__m256i *) &frame1[row+1][col-2+j]);
         __m256i x5=_mm256_loadu_si256((__m256i *) &frame1[row+2][col-2+j]);
         __m256i result1 = _mm256_maddubs_epi16(x1,mask1);
         __m256i result2 = _mm256_maddubs_epi16(x2,mask2);
         __m256i result3 = _mm256_maddubs_epi16(x3,mask3);
         __m256i result4 = _mm256_maddubs_epi16(x4,mask4);
         __m256i result5 = _mm256_maddubs_epi16(x5,mask5);
         __m256i resulk1 = _mm256_maddubs_epi16(x1,masl1);
         __m256i resulk2 = _mm256_maddubs_epi16(x2,masl2);
         __m256i resulk3 = _mm256_maddubs_epi16(x3,masl3);
         __m256i resulk4 = _mm256_maddubs_epi16(x4,masl4);
         __m256i resulk5 = _mm256_maddubs_epi16(x5,masl5);
         __m256i final1 = _mm256_add_epi16(result1,result2);
         __m256i final2 = _mm256_add_epi16(result3,result4);
         __m256i final3 = _mm256_add_epi16(final2,result5);
         __m256i finalk1 = _mm256_add_epi16(resulk1,resulk2);
         __m256i finalk2 = _mm256_add_epi16(resulk3,resulk4);
         __m256i finalk3 = _mm256_add_epi16(finalk2,resulk5);
        
        filt[row][col+j]= (_mm256_extract_epi16(final3,0)+_mm256_extract_epi16(final3,1)+_mm256_extract_epi16(final3,2))/159;
        filt[row][col+5+j]=(_mm256_extract_epi16(finalk3,2)+_mm256_extract_epi16(final3,3)+_mm256_extract_epi16(final3,4))/159;
        filt[row][col+10+j]=(_mm256_extract_epi16(final3,5)+_mm256_extract_epi16(final3,6)+_mm256_extract_epi16(final3,7))/159;
        filt[row][col+15+j]=(_mm256_extract_epi16(finalk3,7)+_mm256_extract_epi16(final3,8)+_mm256_extract_epi16(final3,9))/159;
        filt[row][col+20+j]=(_mm256_extract_epi16(final3,10)+_mm256_extract_epi16(final3,11)+_mm256_extract_epi16(final3,12))/159;
        filt[row][col+25+j]=(_mm256_extract_epi16(finalk3,12)+_mm256_extract_epi16(final3,13)+_mm256_extract_epi16(final3,14))/159;
        

}
            
           
        }
    }
}
 clock_gettime(CLOCK_MONOTONIC, &end);

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);
for (i=(core*256);i<((core+1)*256);i++)
 for (j=0;j<M;j++)
  print[i][j]=filt[i][j];


/*---------------------------- Sobel - Determine edge directions and gradient strengths -------------------------------------------*/
	
}


/*
void Gaussian_Blur_default_unrolled() {

    short int row, col;
    short int newPixel;

    for (row = 2; row < N - 2; row++) {
        for (col = 2; col < M - 2; col++) {
            newPixel = 0;

            newPixel += in_image[row - 2][col - 2] * gaussianMask[0][0];
            newPixel += in_image[row - 2][col - 1] * gaussianMask[0][1];
            newPixel += in_image[row - 2][col] * gaussianMask[0][2];
            newPixel += in_image[row - 2][col + 1] * gaussianMask[0][3];
            newPixel += in_image[row - 2][col + 2] * gaussianMask[0][4];

            newPixel += in_image[row - 1][col - 2] * gaussianMask[1][0];
            newPixel += in_image[row - 1][col - 1] * gaussianMask[1][1];
            newPixel += in_image[row - 1][col] *  gaussianMask[1][2];
            newPixel += in_image[row - 1][col + 1] * gaussianMask[1][3];
            newPixel += in_image[row - 1][col + 2] * gaussianMask[1][4];

            newPixel += in_image[row][col - 2] * gaussianMask[2][0];
            newPixel += in_image[row][col - 1] * gaussianMask[2][1];
            newPixel += in_image[row][col] * gaussianMask[2][2];
            newPixel += in_image[row][col + 1] * gaussianMask[2][3];
            newPixel += in_image[row][col + 2] * gaussianMask[2][4];

            newPixel += in_image[row + 1][col - 2] * gaussianMask[3][0];
            newPixel += in_image[row + 1][col - 1] * gaussianMask[3][1];
            newPixel += in_image[row + 1][col] * gaussianMask[3][2];
            newPixel += in_image[row + 1][col + 1] * gaussianMask[3][3];
            newPixel += in_image[row + 1][col + 2] * gaussianMask[3][4];

            newPixel += in_image[row + 2][col - 2] * gaussianMask[4][0];
            newPixel += in_image[row + 2][col - 1] * gaussianMask[4][1];
            newPixel += in_image[row + 2][col] * gaussianMask[4][2];
            newPixel += in_image[row + 2][col + 1] * gaussianMask[4][3];
            newPixel += in_image[row + 2][col + 2] * gaussianMask[4][4];

            filt_image[row][col] = newPixel / 159;


        }
    }

} 
*/

/*

__m256i maskk = _mm256_set_epi8(gaussianMask[0][0],gaussianMask[0][1],gaussianMask[0][2],gaussianMask[0][3],gaussianMask[0][4],gaussianMask[1][0],gaussianMask[1][1],gaussianMask[1][2],gaussianMask[1][3],gaussianMask[1][4],gaussianMask[2][0],gaussianMask[2][1],gaussianMask[2][2],gaussianMask[2][3],gaussianMask[2][4],gaussianMask[3][0],gaussianMask[3][1],gaussianMask[3][2],gaussianMask[3][3],gaussianMask[3][4],gaussianMask[4][0],gaussianMask[4][1],gaussianMask[4][2],gaussianMask[4][3],gaussianMask[4][4],0,0,0,0,0,0,0);
__m256i result0 = _mm256_setzero_si256();
gas=0;

            __m256i fram = _mm256_set_epi8(frame1[row - 2][col - 2],frame1[row - 2][col - 1],frame1[row - 2][col],frame1[row - 2][col + 1],frame1[row - 2][col + 2],frame1[row - 2][col - 2],frame1[row - 2][col - 1],frame1[row - 2][col],frame1[row - 2][col + 1],frame1[row - 2][col + 2],frame1[row - 2][col - 2],frame1[row - 2][col - 1],frame1[row - 2][col],frame1[row - 2][col + 1],frame1[row - 2][col + 2],frame1[row - 2][col - 2],frame1[row - 2][col - 1],frame1[row - 2][col],frame1[row - 2][col + 1],frame1[row - 2][col + 2],frame1[row - 2][col - 2],frame1[row - 2][col - 1],frame1[row - 2][col],frame1[row - 2][col + 1],frame1[row - 2][col + 2],0,0,0,0,0,0,0);
            
            __m256i resultss = _mm256_maddubs_epi16(fram,maskk);
            __m256i temp1 = _mm256_hadd_epi16(resultss,result0);
            __m256i temp2 = _mm256_hadd_epi16(temp1,result0);
            __m256i temp3 = _mm256_hadd_epi16(temp2,result0);
            gas+= _mm256_extract_epi16(temp3,0);
            gas+= _mm256_extract_epi16(temp3,8);
                        
            filt[row][col] = gas / 159;
*/

/*            newPixel=0;
            newPixel += frame1[row - 2][col - 2] * gaussianMask[0][0];
            newPixel += frame1[row - 2][col - 1] * gaussianMask[0][1];
            newPixel += frame1[row - 2][col] * gaussianMask[0][2];
            newPixel += frame1[row - 2][col + 1] * gaussianMask[0][3];
            newPixel += frame1[row - 2][col + 2] * gaussianMask[0][4];

            newPixel += frame1[row - 1][col - 2] * gaussianMask[1][0];
            newPixel += frame1[row - 1][col - 1] * gaussianMask[1][1];
            newPixel += frame1[row - 1][col] *  gaussianMask[1][2];
            newPixel += frame1[row - 1][col + 1] * gaussianMask[1][3];
            newPixel += frame1[row - 1][col + 2] * gaussianMask[1][4];

            newPixel += frame1[row][col - 2] * gaussianMask[2][0];
            newPixel += frame1[row][col - 1] * gaussianMask[2][1];
            newPixel += frame1[row][col] * gaussianMask[2][2];
            newPixel += frame1[row][col + 1] * gaussianMask[2][3];
            newPixel += frame1[row][col + 2] * gaussianMask[2][4];

            newPixel += frame1[row + 1][col - 2] * gaussianMask[3][0];
            newPixel += frame1[row + 1][col - 1] * gaussianMask[3][1];
            newPixel += frame1[row + 1][col] * gaussianMask[3][2];
            newPixel += frame1[row + 1][col + 1] * gaussianMask[3][3];
            newPixel += frame1[row + 1][col + 2] * gaussianMask[3][4];

            newPixel += frame1[row + 2][col - 2] * gaussianMask[4][0];
            newPixel += frame1[row + 2][col - 1] * gaussianMask[4][1];
            newPixel += frame1[row + 2][col] * gaussianMask[4][2];
            newPixel += frame1[row + 2][col + 1] * gaussianMask[4][3];
            newPixel += frame1[row + 2][col + 2] * gaussianMask[4][4];
            filt[row][col] = newPixel / 159;
            */