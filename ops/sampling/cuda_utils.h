#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H
#include <cmath>
#include <algorithm>

#define TOTAL_THREADS 512
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


inline int opt_n_threads(int work_size)
{
    // round work_size to power of 2 betwwen 512 and 1
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y)
{
    const int x_threads = opt_n_threads(x);
    const int y_threads =
        std::max(std::min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);

    return block_config;
}

#define CUDA_CHECK_ERRORS()                                                 \
    do                                                                      \
    {                                                                       \
        cudaError_t err = cudaGetLastError();                               \
        if (cudaSuccess != err)                                             \
        {                                                                   \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
                    cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
                    __FILE__);                                              \
            exit(-1);                                                       \
        }                                                                   \
    } while (0)
#endif