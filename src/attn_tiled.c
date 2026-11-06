/*
 * attn_tiled.c -- Q3b: cache and register tiling.
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

static void tiled_gemm(const float *restrict A, const float *restrict B,
                       float *restrict C, int M, int P, int R)
{
    memset(C, 0, (size_t)M * P * sizeof(float));
    for (int ii = 0; ii < M; ii += 32)
        for (int jj = 0; jj < P; jj += 64)
            for (int kk = 0; kk < R; kk += 64) {
                const int ie = ii + 32 < M ? ii + 32 : M;
                const int je = jj + 64 < P ? jj + 64 : P;
                const int ke = kk + 64 < R ? kk + 64 : R;
                for (int i = ii; i < ie; i += 4)
                    for (int j = jj; j < je; j += 8) {
                        const int mr = ie - i < 4 ? ie - i : 4;
                        const int nr = je - j < 8 ? je - j : 8;
                        if (mr == 4 && nr == 8) {
                            float a[4][8];
                            for (int r = 0; r < 4; ++r)
                                for (int c = 0; c < 8; ++c)
                                    a[r][c] = C[(size_t)(i+r)*P+j+c];
                            for (int k = kk; k < ke; ++k) {
                                #pragma clang loop unroll(full)
                                for (int r = 0; r < 4; ++r) {
                                    const float x = A[(size_t)(i+r)*R+k];
                                    #pragma clang loop unroll(full)
                                    for (int c = 0; c < 8; ++c)
                                        a[r][c] += x * B[(size_t)k*P+j+c];
                                }
                            }
                            for (int r = 0; r < 4; ++r)
                                for (int c = 0; c < 8; ++c)
                                    C[(size_t)(i+r)*P+j+c] = a[r][c];
                            continue;
                        }
                        float acc[4][8] = {{0}};
                        for (int r = 0; r < mr; ++r)
                            for (int c = 0; c < nr; ++c)
                                acc[r][c] = C[(size_t)(i+r)*P+j+c];
                        for (int k = kk; k < ke; ++k)
                            for (int r = 0; r < mr; ++r) {
                                const float a = A[(size_t)(i+r)*R+k];
                                for (int c = 0; c < nr; ++c)
                                    acc[r][c] += a * B[(size_t)k*P+j+c];
                            }
                        for (int r = 0; r < mr; ++r)
                            for (int c = 0; c < nr; ++c)
                                C[(size_t)(i+r)*P+j+c] = acc[r][c];
                    }
            }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    float *T = transpose_keys(K, N, d);
    float *S = checked_malloc((size_t)N * N * sizeof(float));
    const float scale = 1.0f / sqrtf((float)d);
    tiled_gemm(Q, T, S, N, N, d);
    for (size_t x = 0; x < (size_t)N * N; ++x) S[x] *= scale;
    softmax_rows(S, N, N);
    tiled_gemm(S, V, O, N, d, N);
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
