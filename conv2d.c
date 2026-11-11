// Compile: gcc -O3 -mavx2 -o conv2d conv2d.c

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdalign.h>



// Input height, width, filter height width and output height, width defined below

#define IN_H 2048   
#define IN_W 2048


#define K_H 4
#define K_W 4

typedef int8_t  IN_DT;  // Filter and Input should be of same data type for vector operation

#define OUT_H (IN_H - K_H + 1)
#define OUT_W (IN_W - K_W + 1)
typedef int32_t OUT_DT;


// Set exactly one of these to 1; set the others to 0.
// Example: USE_AVX2=1 means compile and run AVX2 version.
#define USE_SSE 0
#define USE_AVX2 1
#define USE_AVX512 0


void fill_random_int8(IN_DT *arr, int size, int min, int max) {
    for (int i = 0; i < size; ++i)
        arr[i] = (IN_DT)(min + rand() % (max - min + 1));
}


// Naive convolution
void conv2d_naive(
    const IN_DT *input, const IN_DT *kernel,
    OUT_DT *output,
    int in_h, int in_w, int k_h, int k_w
) {
    for (int i = 0; i < in_h - k_h + 1; ++i) {
        for (int j = 0; j < in_w - k_w + 1; ++j) {
            OUT_DT sum = 0;
            for (int ki = 0; ki < k_h; ++ki) {
                for (int kj = 0; kj < k_w; ++kj) {
                    OUT_DT val = input[(i + ki) * in_w + (j + kj)];
                    OUT_DT fval = kernel[ki * k_w + kj];
                    sum += val * fval;
                }
            }
            output[i * (in_w - k_w + 1) + j] = sum;
        }
    }
}


#if   USE_AVX512
// Build: g++ -O3 -mavx512f -mavx512bw conv2d.c -o conv_avx512
void conv2d_avx512(
    const IN_DT *input, const IN_DT *kernel,
    OUT_DT *output,
    int in_h, int in_w, int k_h, int k_w
){
    const int oh = in_h - k_h + 1;
    const int ow = in_w - k_w + 1;

    for (int i = 0; i < oh; ++i){
        for (int j = 0; j < ow; ++j){
            OUT_DT sum = 0;

            for (int ki = 0; ki < k_h; ++ki){
                if (k_w == 4) {
                    // --- Fast path: load EXACTLY 4 bytes, zero the rest ---
                    // Place 4 input bytes into a 256-bit reg (lower 32 bits used)
                    //use _mm256_cvtsi32_si256 to load one for each in and kernel
                    

                    // AVX-512BW: sign-extend 32x i8 -> 32x i16 (in a 512-bit reg)
                    // use _mm512_cvtepi8_epi16 for both in and kernel
                    __m512i in16  = _mm512_cvtepi8_epi16(in8_32);
                    __m512i ker16 = _mm512_cvtepi8_epi16(ker8_32);

                    // 32-lane int16 multiply
                    // use _mm512_mullo_epi16 to multiply
                    

                    // Sum only the first 4 lanes (the rest are zero)
                    alignas(64) int16_t tmp[32];
                    //use _mm512_storeu_si512 to store the result back to tmp
                    
                    sum += (OUT_DT)tmp[0] + (OUT_DT)tmp[1]
                         + (OUT_DT)tmp[2] + (OUT_DT)tmp[3];
                } else {
                    // --- Simple fallback: your original scalar accumulate ---
                    for (int c = 0; c < k_w; ++c) {
                        sum += (OUT_DT)input[(i+ki)*in_w + (j+c)]
                             * (OUT_DT)kernel[ki*k_w + c];
                    }
                }
            }
            output[i*ow + j] = sum;
        }
    }
}
#elif USE_AVX2
// Build: g++ -O3 -mavx2 -mfma conv2d.c -o conv_avx2
void conv2d_avx2(
    const IN_DT *input, const IN_DT *kernel,
    OUT_DT *output,
    int in_h, int in_w, int k_h, int k_w
){
    const int oh = in_h - k_h + 1;
    const int ow = in_w - k_w + 1;

    for (int i = 0; i < oh; ++i){
        for (int j = 0; j < ow; ++j){
            OUT_DT sum = 0;

            for (int ki = 0; ki < k_h; ++ki){
                if (k_w == 4) {
                    // Load EXACTLY 4 bytes (no row over-read), into 128-bit reg
                    //use _mm_cvtsi32_si128 to load one for each in and kernel
                    

                    int32_t in32, ker32;

                    memcpy(&in32, &input[(i + ki) * in_w + j], 4);
                    memcpy(&ker32, &kernel[ki * k_w], 4);

                    __m128i in8 = _mm_cvtsi32_si128(in32);
                    __m128i ker8 = _mm_cvtsi32_si128(ker32);


                    // Widen int8 -> 16 lanes of int16 in a 256-bit reg
                    // use _mm256_cvtepi8_epi16 for both in and kernel
                    __m256i in16  = _mm256_cvtepi8_epi16(in8);
                    __m256i ker16 = _mm256_cvtepi8_epi16(ker8);

                    // Elementwise multiply: 16x int16
                    // use _mm256_mullo_epi16 to multiply

                    __m256i prod = _mm256_mullo_epi16(in16, ker16);
                    

                    // Sum first 4 lanes (the ones we actually used)
                    alignas(32) int16_t tmp[16];
                    //use _mm256_storeu_si256 to store the result back to tmp
                    _mm256_storeu_si256((__m256i*)tmp, prod);
                    
                    sum += (OUT_DT)tmp[0] + (OUT_DT)tmp[1]
                         + (OUT_DT)tmp[2] + (OUT_DT)tmp[3];
                } else {
                    // Simple fallback: your original scalar accumulate
                    for (int c = 0; c < k_w; ++c) {
                        sum += (OUT_DT)input[(i+ki)*in_w + (j+c)]
                             * (OUT_DT)kernel[ki*k_w + c];
                    }
                }
            }
            output[i*ow + j] = sum;
        }
    }
}
#else
// Build: g++ -O3 -msse4.1 conv2d.c -o conv_sse
void conv2d_sse(
    const IN_DT *input, const IN_DT *kernel,
    OUT_DT *output,
    int in_h, int in_w, int k_h, int k_w
){
    const int oh = in_h - k_h + 1;
    const int ow = in_w - k_w + 1;

    for (int i = 0; i < oh; ++i){
        for (int j = 0; j < ow; ++j){
            OUT_DT sum = 0;

            for (int ki = 0; ki < k_h; ++ki){
                if (k_w == 4) {
                    // Fast path: load EXACTLY 4 bytes (no row over-read)
                    //use _mm_cvtsi32_si128 to load one for each in and kernel
                    

                    //use _mm_cvtepi8_epi16 for int 8 to int16 conversion for input and kernel
                    
                    //use _mm_mullo_epi16 to multiply in and kernel
                    

                    alignas(16) int16_t tmp[8];
                    //use _mm_storeu_si128 to store the result back to tmp
                    
                    sum += (OUT_DT)tmp[0] + (OUT_DT)tmp[1]
                         + (OUT_DT)tmp[2] + (OUT_DT)tmp[3];
                } else {
                    // Simple fallback: keep your original scalar accumulate
                    for (int c = 0; c < k_w; ++c) {
                        sum += (OUT_DT)input[(i+ki)*in_w + (j+c)]
                             * (OUT_DT)kernel[ki*k_w + c];
                    }
                }
            }
            output[i*ow + j] = sum;
        }
    }
}
#endif


// Utility: compare outputs
int compare_outputs(const OUT_DT *a, const OUT_DT *b, int sz) {
    for (int i = 0; i < sz; ++i)
        if (a[i] != b[i]) return 0;
    return 1;
}

int main() {
    srand(time(NULL));
    // Allocate
    IN_DT *input = (IN_DT*)malloc(IN_H * IN_W * sizeof(IN_DT));
    IN_DT *kernel = (IN_DT*)malloc(K_H * K_W * sizeof(IN_DT));
    OUT_DT *output_naive = (OUT_DT*)malloc(OUT_H * OUT_W * sizeof(OUT_DT));
    OUT_DT *output_avx = (OUT_DT*)malloc(OUT_H * OUT_W * sizeof(OUT_DT));

    fill_random_int8(input, IN_H * IN_W, -4, 4);  // avoid overflow
    fill_random_int8(kernel, K_H * K_W, -2, 2);   // tiny numbers

    // Timing naive
    clock_t t0 = clock();
    conv2d_naive(input, kernel, output_naive, IN_H, IN_W, K_H, K_W);
    clock_t t1 = clock();
    double time_naive = (double)(t1 - t0) / CLOCKS_PER_SEC;

    // Timing AVX
    t0 = clock();
    
    #if   USE_AVX512
        conv2d_avx512(input, kernel, output_avx, IN_H, IN_W, K_H, K_W);
    #elif USE_AVX2
        conv2d_avx2(input, kernel, output_avx, IN_H, IN_W, K_H, K_W);
    #elif USE_SSE
        conv2d_sse(input, kernel, output_avx, IN_H, IN_W, K_H, K_W);
    #endif
    

    t1 = clock();
    double time_avx = (double)(t1 - t0) / CLOCKS_PER_SEC;

    // Check
    printf("Correct: %d\n", compare_outputs(output_naive, output_avx, OUT_H * OUT_W));
    printf("Naive: %.3f sec\n", time_naive);
    printf("AVX:   %.3f sec\n", time_avx);

    free(input);
    free(kernel);
    free(output_naive);
    free(output_avx);
    return 0;
}










