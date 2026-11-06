/*
 * attn_best.c -- Q3d: your fastest single-threaded version. Use any technique you like.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */

static void softmax_rows(float *S, int rows, int N)
{
    for (int i = 0; i < rows; ++i) {
        float *row = S + (size_t)i * N;
        float maximum = row[0];
        for (int j = 1; j < N; ++j)
            if (row[j] > maximum) maximum = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            row[j] = expf(row[j] - maximum);
            sum += row[j];
        }
        for (int j = 0; j < N; ++j) row[j] /= sum;
    }
}

static void *checked_malloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "attention: allocation failed\n"); exit(EXIT_FAILURE); }
    return p;
}

static float *transpose_keys(const float *K, int N, int d)
{
    float *T = checked_malloc((size_t)N * d * sizeof(float));
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < d; ++k)
            T[(size_t)k * N + j] = K[(size_t)j * d + k];
    return T;
}

#include <arm_neon.h>

static void neon_gemm(const float *restrict A, const float *restrict B, float *restrict C, int M, int P, int R)
{
    int i = 0;
    for (; i + 3 < M; i += 4) {
        int j = 0;
        for (; j + 15 < P; j += 16) {
            float32x4_t c0_0 = vdupq_n_f32(0);
            float32x4_t c0_1 = vdupq_n_f32(0);
            float32x4_t c0_2 = vdupq_n_f32(0);
            float32x4_t c0_3 = vdupq_n_f32(0);
            float32x4_t c1_0 = vdupq_n_f32(0);
            float32x4_t c1_1 = vdupq_n_f32(0);
            float32x4_t c1_2 = vdupq_n_f32(0);
            float32x4_t c1_3 = vdupq_n_f32(0);
            float32x4_t c2_0 = vdupq_n_f32(0);
            float32x4_t c2_1 = vdupq_n_f32(0);
            float32x4_t c2_2 = vdupq_n_f32(0);
            float32x4_t c2_3 = vdupq_n_f32(0);
            float32x4_t c3_0 = vdupq_n_f32(0);
            float32x4_t c3_1 = vdupq_n_f32(0);
            float32x4_t c3_2 = vdupq_n_f32(0);
            float32x4_t c3_3 = vdupq_n_f32(0);
            for (int k = 0; k < R; ++k) {
                float32x4_t b0 = vld1q_f32(B + (size_t)k*P+j+0);
                float32x4_t b1 = vld1q_f32(B + (size_t)k*P+j+4);
                float32x4_t b2 = vld1q_f32(B + (size_t)k*P+j+8);
                float32x4_t b3 = vld1q_f32(B + (size_t)k*P+j+12);
                float a0 = A[(size_t)(i+0)*R+k];
                c0_0 = vfmaq_n_f32(c0_0, b0, a0);
                c0_1 = vfmaq_n_f32(c0_1, b1, a0);
                c0_2 = vfmaq_n_f32(c0_2, b2, a0);
                c0_3 = vfmaq_n_f32(c0_3, b3, a0);
                float a1 = A[(size_t)(i+1)*R+k];
                c1_0 = vfmaq_n_f32(c1_0, b0, a1);
                c1_1 = vfmaq_n_f32(c1_1, b1, a1);
                c1_2 = vfmaq_n_f32(c1_2, b2, a1);
                c1_3 = vfmaq_n_f32(c1_3, b3, a1);
                float a2 = A[(size_t)(i+2)*R+k];
                c2_0 = vfmaq_n_f32(c2_0, b0, a2);
                c2_1 = vfmaq_n_f32(c2_1, b1, a2);
                c2_2 = vfmaq_n_f32(c2_2, b2, a2);
                c2_3 = vfmaq_n_f32(c2_3, b3, a2);
                float a3 = A[(size_t)(i+3)*R+k];
                c3_0 = vfmaq_n_f32(c3_0, b0, a3);
                c3_1 = vfmaq_n_f32(c3_1, b1, a3);
                c3_2 = vfmaq_n_f32(c3_2, b2, a3);
                c3_3 = vfmaq_n_f32(c3_3, b3, a3);
            }
            vst1q_f32(C + (size_t)(i+0)*P+j+0, c0_0);
            vst1q_f32(C + (size_t)(i+0)*P+j+4, c0_1);
            vst1q_f32(C + (size_t)(i+0)*P+j+8, c0_2);
            vst1q_f32(C + (size_t)(i+0)*P+j+12, c0_3);
            vst1q_f32(C + (size_t)(i+1)*P+j+0, c1_0);
            vst1q_f32(C + (size_t)(i+1)*P+j+4, c1_1);
            vst1q_f32(C + (size_t)(i+1)*P+j+8, c1_2);
            vst1q_f32(C + (size_t)(i+1)*P+j+12, c1_3);
            vst1q_f32(C + (size_t)(i+2)*P+j+0, c2_0);
            vst1q_f32(C + (size_t)(i+2)*P+j+4, c2_1);
            vst1q_f32(C + (size_t)(i+2)*P+j+8, c2_2);
            vst1q_f32(C + (size_t)(i+2)*P+j+12, c2_3);
            vst1q_f32(C + (size_t)(i+3)*P+j+0, c3_0);
            vst1q_f32(C + (size_t)(i+3)*P+j+4, c3_1);
            vst1q_f32(C + (size_t)(i+3)*P+j+8, c3_2);
            vst1q_f32(C + (size_t)(i+3)*P+j+12, c3_3);
        }
        for (; j < P; ++j) for (int r = 0; r < 4; ++r) {
            float sum = 0;
            for (int k = 0; k < R; ++k) sum += A[(size_t)(i+r)*R+k]*B[(size_t)k*P+j];
            C[(size_t)(i+r)*P+j] = sum;
        }
    }
    for (; i < M; ++i) for (int j = 0; j < P; ++j) {
        float sum = 0;
        for (int k = 0; k < R; ++k) sum += A[(size_t)i*R+k]*B[(size_t)k*P+j];
        C[(size_t)i*P+j] = sum;
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    enum { QUERY_BLOCK = 32 };
    float *T = transpose_keys(K, N, d);
    float *S = checked_malloc((size_t)QUERY_BLOCK * N * sizeof(float));
    const float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < N; i += QUERY_BLOCK) {
        const int rows = N - i < QUERY_BLOCK ? N - i : QUERY_BLOCK;
        neon_gemm(Q + (size_t)i*d, T, S, rows, N, d);
        for (size_t x = 0; x < (size_t)rows*N; ++x) S[x] *= scale;
        softmax_rows(S, rows, N);
        neon_gemm(S, V, O + (size_t)i*d, rows, d, N);
    }
    free(S);
    free(T);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
