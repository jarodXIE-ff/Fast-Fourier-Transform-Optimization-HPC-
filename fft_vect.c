// #define  _POSIX_C_SOURCE 1

// XIE JAROD 28710097

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;

double cutoff = 500;
u64 seed = 0;
u64 size = 0;
char *filename = NULL;
int version = 1;

typedef struct arrayComplex_
{
    double *real;
    double *imag;
} arrayComplex;

/******************** pseudo-random function (SPECK-like) ********************/

#define ROR(x, r) ((x >> r) | (x << (64 - r)))
#define ROL(x, r) ((x << r) | (x >> (64 - r)))
#define R(x, y, k) (x = ROR(x, 8), x += y, x ^= k, y = ROL(y, 3), y ^= x)
u64 PRF(u64 seed, u64 IV, u64 i)
{
    u64 y = i;
    u64 x = 0xBaadCafeDeadBeefULL;
    u64 b = IV;
    u64 a = seed;
    R(x, y, b);
    for (int i = 0; i < 32; i++)
    {
        R(a, b, i);
        R(x, y, b);
    }
    return x + i;
}

/************************** Fast Fourier Transform ***************************/
/* This code assumes that n is a power of two !!!                            */
/*****************************************************************************/

void FFT_rec_small_vect(u64 n, const double *XR, const double *XI, double *YR, double *YI, u64 stride)
{
    /* Same as FFT_rec_vect without tasking. */
    if (n == 1)
    {
        YR[0] = XR[0];
        YI[0] = XI[0];
        return;
    }

    double complex omega_n = cexp(-2 * I * M_PI / n); /* n-th root of unity*/
    double complex omega = 1;
    double omegaR = creal(omega);
    double omegaI = cimag(omega); /* twiddle factor */
    double omega_nR = creal(omega_n);
    double omega_nI = cimag(omega_n);

    FFT_rec_small_vect(n / 2, XR, XI, YR, YI, 2 * stride);
    FFT_rec_small_vect(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride);

    for (u64 i = 0; i < n / 2; i++)
    {
        // Les lignes suivantes sont équivalentes à chaque lignes de FFT_rec_small sans vectorisation
        double pReal = YR[i];
        double pImg = YI[i];

        double qReal = (YR[i + n / 2] * omegaR) - (YI[i + n / 2] * omegaI);
        double qImg = (YR[i + n / 2] * omegaI) + (YI[i + n / 2] * omegaR);

        YR[i] = pReal + qReal;
        YI[i] = pImg + qImg;

        YR[i + n / 2] = pReal - qReal;
        YI[i + n / 2] = pImg - qImg;

        double tmpR = omegaR * omega_nR - omegaI * omega_nI;
        double tmpI = omegaR * omega_nI + omegaI * omega_nR;
        omegaR = tmpR;
        omegaI = tmpI;
    }
}

void FFT_rec_vect(u64 n, const double *XR, const double *XI, double *YR, double *YI, u64 stride)
{
    /* V1: Without powers_omega in parameter. */
    if (n == 1)
    {
        YR[0] = XR[0];
        YI[0] = XI[0];
        return;
    }

    double complex omega_n = cexp(-2 * I * M_PI / n); /* n-th root of unity*/
    double complex omega = 1;                         /* twiddle factor */
    double omegaR = creal(omega);
    double omegaI = cimag(omega);
    double omega_nR = creal(omega_n);
    double omega_nI = cimag(omega_n);

    if (n < 2048)
    {
        FFT_rec_small_vect(n / 2, XR, XI, YR, YI, 2 * stride);
        FFT_rec_small_vect(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride);
    }
    else
    {
#pragma omp task
        FFT_rec_vect(n / 2, XR, XI, YR, YI, 2 * stride);
#pragma omp task
        FFT_rec_vect(n / 2, XR + stride, XI + stride, YR + n / 2, YI + n / 2, 2 * stride);
#pragma omp taskwait
    }

    for (u64 i = 0; i < n / 2; i++)
    {
        // Les lignes suivantes sont équivalentes à chaque lignes de FFT_rec sans vectorisation
        double pReal = YR[i];
        double pImg = YI[i];

        double qReal = (YR[n / 2 + i] * omegaR) - (YI[n / 2 + i] * omegaI);
        double qImg = (YR[n / 2 + i] * omegaI) + (YI[n / 2 + i] * omegaR);

        YR[i] = pReal + qReal;
        YI[i] = pImg + qImg;

        YR[n / 2 + i] = pReal - qReal;
        YI[n / 2 + i] = pImg - qImg;

        double tmpR = omegaR * omega_nR - omegaI * omega_nI;
        double tmpI = omegaR * omega_nI + omegaI * omega_nR;
        omegaR = tmpR;
        omegaI = tmpI;
    }
}

void FFT_vect(u64 n, const arrayComplex *X, arrayComplex *Y)
{
    /* FFT: Recursive without the powers of omega in parameters. */

    if ((n & (n - 1)) != 0)
    {
        errx(1, "size is not a power of two (this code does not handle other cases)");
    }

#pragma omp parallel
    {
#pragma omp single
        FFT_rec_vect(n, X->real, X->imag, Y->real, Y->imag, 1); /* stride == 1 initially */
    }
}

/* Computes the inverse Fourier transform, but destroys the input */
void iFFT_vect(u64 n, arrayComplex *X, arrayComplex *Y)
{
    __m256d vect_minus_one = _mm256_set1_pd(-1.0);

#pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d vect_imag = _mm256_load_pd(&X->imag[i]);
        __m256d vect_res = _mm256_mul_pd(vect_minus_one, vect_imag);
        _mm256_store_pd(&X->imag[i], vect_res);
    }

    FFT_vect(n, X, Y);

    __m256d div_n = _mm256_set1_pd(n);

#pragma omp parallel for
    for (u64 i = 0; i < n; i += 4)
    {
        __m256d real_vals = _mm256_load_pd(&Y->real[i]);
        __m256d imag_vals = _mm256_load_pd(&Y->imag[i]);

        real_vals = _mm256_div_pd(real_vals, div_n);
        imag_vals = _mm256_div_pd(imag_vals, div_n);
        imag_vals = _mm256_mul_pd(imag_vals, vect_minus_one);

        _mm256_store_pd(&Y->real[i], real_vals);
        _mm256_store_pd(&Y->imag[i], imag_vals);
    }
}

/******************* utility functions ********************/

void storeTime(double totalTime, const char *filename)
{
    FILE *file = fopen(filename, "a");

    if (file == NULL)
    {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s.\n", filename);
        return;
    }

    fprintf(file, "%f secondes\n", totalTime);

    fclose(file);
}

double wtime()
{
    struct timeval ts;
    gettimeofday(&ts, NULL);
    return (double)ts.tv_sec + ts.tv_usec / 1e6;
}

void process_command_line_options(int argc, char **argv)
{
    struct option longopts[6] = {
        {"size", required_argument, NULL, 'n'},
        {"seed", required_argument, NULL, 's'},
        {"output", required_argument, NULL, 'o'},
        {"cutoff", required_argument, NULL, 'c'},
        {"version", required_argument, NULL, 'v'},
        {NULL, 0, NULL, 0}};

    char ch;
    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1)
    {
        switch (ch)
        {
        case 'n':
            size = atoll(optarg);
            break;
        case 's':
            seed = atoll(optarg);
            break;
        case 'o':
            filename = optarg;
            break;
        case 'c':
            cutoff = atof(optarg);
            break;
        case 'v':
            version = atoi(optarg);
            break;
        default:
            errx(1, "Unknown option\n");
        }
    }
    /* validation */
    if (size == 0)
        errx(1, "missing --size argument");
}

/* save at most 10s of sound output in .WAV format */
void save_WAV(char *filename, u64 size, arrayComplex *C)
{

    assert(size < 1000000000);
    FILE *f = fopen(filename, "w");
    if (f == NULL)
        err(1, "fopen");

    printf("Writing <= 10s of audio output in %s\n", filename);
    u32 rate = 44100; // Sample rate
    u32 frame_count = 10 * rate;
    if (size < frame_count)
        frame_count = size;
    u16 chan_num = 2; // Number of channels
    u16 bits = 16;    // Bit depth
    u32 length = frame_count * chan_num * bits / 8;
    u16 byte;
    double multiplier = 32767;

    /* WAVE Header Data */
    fwrite("RIFF", 1, 4, f);
    u32 chunk_size = length + 44 - 8;
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    u32 subchunk1_size = 16;
    fwrite(&subchunk1_size, 4, 1, f);
    u16 fmt_type = 1; // 1 = PCM
    fwrite(&fmt_type, 2, 1, f);
    fwrite(&chan_num, 2, 1, f);
    fwrite(&rate, 4, 1, f);

    // (Sample Rate * BitsPerSample * Channels) / 8
    uint32_t byte_rate = rate * bits * chan_num / 8;
    fwrite(&byte_rate, 4, 1, f);
    uint16_t block_align = chan_num * bits / 8;
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);

    /* Marks the start of the data */
    fwrite("data", 1, 4, f);
    fwrite(&length, 4, 1, f); // Data size
    for (u32 i = 0; i < frame_count; i++)
    {
        byte = C->real[i] * multiplier;
        fwrite(&byte, 2, 1, f);
        byte = C->imag[i] * multiplier;
        fwrite(&byte, 2, 1, f);
    }

    fclose(f);
}

/*************************** main function *********************************/

int main(int argc, char **argv)
{
    process_command_line_options(argc, argv);
    printf("Nb thread = %d \n", omp_get_max_threads());

    /* generate white noise */
    arrayComplex *A = malloc(sizeof(*A));
    arrayComplex *B = malloc(sizeof(*B));
    arrayComplex *C = malloc(sizeof(*C));

    A->real = (double *)aligned_alloc(32, size * sizeof(double));
    A->imag = (double *)aligned_alloc(32, size * sizeof(double));

    B->real = (double *)aligned_alloc(32, size * sizeof(double));
    B->imag = (double *)aligned_alloc(32, size * sizeof(double));

    C->real = (double *)aligned_alloc(32, size * sizeof(double));
    C->imag = (double *)aligned_alloc(32, size * sizeof(double));

    printf("Generating white noise...\n");
    double start = wtime();

#pragma omp parallel for
    for (u64 i = 0; i < size; i++)
    {
        double real = 2 * (PRF(seed, 0, i) * 5.42101086242752217e-20) - 1;
        double imag = 2 * (PRF(seed, 1, i) * 5.42101086242752217e-20) - 1;
        A->real[i] = real;
        A->imag[i] = imag;
    }

    printf("Forward FFT...\n");
    FFT_vect(size, A, B);

    /* damp fourrier coefficients */
    printf("Adjusting Fourier coefficients...\n");

#pragma omp parallel for
    for (u64 i = 0; i < size; i++)
    {
        double tmp = sin(i * 2 * M_PI / 44100);

        double cexp_real = cos(-i * 2 * M_PI / 4 / 44100);
        double cexp_imag = sin(-i * 2 * M_PI / 4 / 44100);

        double tmpBreal = (B->real[i] * cexp_real - B->imag[i] * cexp_imag) * tmp;
        B->imag[i] = (B->real[i] * cexp_imag + B->imag[i] * cexp_real) * tmp;
        B->real[i] = tmpBreal;

        B->real[i] *= (i + 1) / exp((i * cutoff) / size);
        B->imag[i] *= (i + 1) / exp((i * cutoff) / size);
    }

    printf("Inverse FFT...\n");
    iFFT_vect(size, B, C);

    printf("Normalizing output...\n");
    double max = 0;

#pragma omp parallel for reduction(max : max)
    for (u64 i = 0; i < size; i++)
    {
        max = fmax(max, sqrt((C->real[i] * C->real[i]) + (C->imag[i] * C->imag[i])));
    }
    printf("max = %g\n", max);

    __m256d inv_max = _mm256_set1_pd(max);
#pragma omp parallel for
    for (u64 i = 0; i < size; i += 4)
    {
        __m256d real_vals = _mm256_load_pd(&C->real[i]);
        __m256d imag_vals = _mm256_load_pd(&C->imag[i]);

        real_vals = _mm256_div_pd(real_vals, inv_max);
        imag_vals = _mm256_div_pd(imag_vals, inv_max);

        _mm256_store_pd(&C->real[i], real_vals);
        _mm256_store_pd(&C->imag[i], imag_vals);
    }

    double end = wtime();
    double time = end - start;

    printf("Temps = %fs\n", time);
    storeTime(time, "FFT_VECT_TIME.txt");

    if (filename != NULL)
        save_WAV(filename, size, C);

    exit(EXIT_SUCCESS);
}