#include <stdio.h>
#include <float.h>
#include <stdint.h>
#include <math.h>
#include <random>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>


#define CSC(call)  					                                \
do {								                                \
	cudaError_t res = call;		                                	\
	if (res != cudaSuccess) {	                                	\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);				                                   	\
	}							                                   	\
} while(0)


#define REDUCE_SHMEM_SIZE 1024
#define BINS_SHMEM_SIZE 2048
#define REDUCE_MIN 0
#define REDUCE_MAX 1 


float global_max = 0;
float global_min = 0;
uint32_t rec_cnt = 0;


__device__ float atIndex(float *arr, uint32_t index, uint32_t len, int fun_type)
{
    if (index >= len) {
        return (fun_type == REDUCE_MIN ? FLT_MAX: -FLT_MAX);
    }
    return arr[index];
}


__global__ void reduce_kernel(float *in, float *out, uint32_t len, int fun_type)
{
    uint32_t tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float (*reduce_fun)(float, float) = (fun_type == REDUCE_MIN ? fminf: fmaxf);
    
    __shared__ float partial_res[REDUCE_SHMEM_SIZE];
	partial_res[threadIdx.x] = reduce_fun(
        atIndex(in, tid, len, fun_type), 
        atIndex(in, tid + blockDim.x, len, fun_type)
    );
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_res[threadIdx.x] = reduce_fun(partial_res[threadIdx.x], partial_res[threadIdx.x + s]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
        out[blockIdx.x] = partial_res[0];
	}
}


float reduce(float *arr, uint32_t len, int fun_type, float *buff = NULL)
{
    if (buff == NULL) {
        buff = arr;
    }

    uint32_t BLOCK_CNT = ceil(double(len) / REDUCE_SHMEM_SIZE / 2);
    reduce_kernel<<<BLOCK_CNT, REDUCE_SHMEM_SIZE>>>(arr, buff, len, fun_type);
    CSC(cudaGetLastError());

    while (BLOCK_CNT > 1) {
        len = BLOCK_CNT;
        BLOCK_CNT = ceil(double(len) / REDUCE_SHMEM_SIZE / 2);
        reduce_kernel<<<BLOCK_CNT, REDUCE_SHMEM_SIZE>>>(buff, buff, len, fun_type);
        CSC(cudaGetLastError());
    }

    float ret[1];
    CSC(cudaMemcpy(ret, buff, sizeof(float), cudaMemcpyDeviceToHost));
    return ret[0];
}


__global__ void histogram_kernel(float *arr, uint32_t len, double min, double max, uint32_t *bins_size, uint32_t nbins) 
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offset = gridDim.x * blockDim.x;
    double range = max - min;

    uint32_t bin_num;
    for (uint32_t i = tid; i < len; i += offset) {
        bin_num = nbins * ((arr[i] - min) / range);
        bin_num = (bin_num < nbins ? bin_num: nbins - 1);
        atomicAdd(&bins_size[bin_num], 1);
    }
}


__global__ void prepare_buckets(float *in, float *out, uint32_t len, double min, double max, uint32_t *bins_start, uint32_t nbins)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offset = gridDim.x * blockDim.x;
    double range = max - min;
    
    float el;
    uint32_t bin_num;
    for (uint32_t i = tid; i < len; i += offset) {
        el = in[i];
        bin_num = nbins * ((el - min) / range);
        bin_num = (bin_num < nbins ? bin_num: nbins - 1);
        out[atomicAdd(&bins_start[bin_num], 1)] = el;
    }
}


__global__ void bucket_bitonic_sort(float *in, float *out, uint32_t len, 
                                    uint32_t *bins_end, uint32_t *bins_size, 
                                    uint32_t *queue_bins_size, uint32_t *queue_bins_start,
                                    uint32_t nbins, uint32_t *gpu_queue_len)
{
    for (uint32_t cur_block_id = blockIdx.x; cur_block_id < nbins; cur_block_id += gridDim.x) 
    {
        uint32_t bin_size =  bins_size[cur_block_id];
        uint32_t bin_start = bins_end[cur_block_id] - bin_size;

        if (bin_size > BINS_SHMEM_SIZE || bin_size <= 0) {
            if (bin_size > BINS_SHMEM_SIZE && threadIdx.x == 0) {
                uint32_t i = atomicAdd(gpu_queue_len, 1);
                queue_bins_size[i] = bin_size;
                queue_bins_start[i] = bin_start;
            }
            __syncthreads();
            continue;
        }
        // blocksDim.x == BINS_SHMEM_SIZE / 2
        __shared__ float buff[BINS_SHMEM_SIZE];

        buff[threadIdx.x] = (threadIdx.x < bin_size ? in[bin_start + threadIdx.x]: FLT_MAX);
        buff[threadIdx.x + blockDim.x] =  ((threadIdx.x + blockDim.x) < bin_size ? in[bin_start + threadIdx.x + blockDim.x]: FLT_MAX);

        for (uint32_t bitonic_size = 2; bitonic_size <= BINS_SHMEM_SIZE; bitonic_size <<= 1) {
            uint32_t dir =  (threadIdx.x / (bitonic_size >> 1)) & 1;
            for (uint32_t hf_stride = (bitonic_size >> 1); hf_stride > 0; hf_stride >>= 1) {
                __syncthreads();

                uint32_t t_index = (threadIdx.x / hf_stride) * hf_stride * 2 + (threadIdx.x & (hf_stride - 1));

                if ((buff[t_index] > buff[t_index + hf_stride]) != dir) {
                    float temp = buff[t_index];
                    buff[t_index] = buff[t_index + hf_stride];
                    buff[t_index + hf_stride] = temp;
                }
            }
        }

        __syncthreads();
        if (threadIdx.x < bin_size) {
            out[bin_start + threadIdx.x] = buff[threadIdx.x];
        }
        if ((threadIdx.x + blockDim.x) < bin_size) {
            out[bin_start + threadIdx.x + blockDim.x] = buff[threadIdx.x + blockDim.x];
        }
        __syncthreads();
    }
}


void recursive_bucket_sort(float *dev_arr, float *buff, uint32_t len,
                           uint32_t *bins_start, uint32_t *bins_size,    
                           uint32_t *gpu_queue_len,
                           uint32_t *queue_bins_start, uint32_t *queue_bins_size)
{
    if (len <= 1) {
        return;
    }

    float max = reduce(dev_arr, len, REDUCE_MAX, buff);
    float min = reduce(dev_arr, len, REDUCE_MIN, buff);

    if (min == max) {
        return;
    }

    uint32_t AVG_BIN_SIZE = BINS_SHMEM_SIZE / 2;
    uint32_t nbins = ceil(float(len) / AVG_BIN_SIZE);
    uint32_t BINS_BLOCK_SIZE = (nbins < 65500 ? nbins: 65500);

    CSC(cudaMemset(bins_size, 0, sizeof(uint32_t) * nbins));
    CSC(cudaMemset(gpu_queue_len, 0, sizeof(uint32_t)));

    // можно быть поэкономнее с размерами сетки?
    histogram_kernel<<<BINS_BLOCK_SIZE, AVG_BIN_SIZE>>>(dev_arr, len, min, max, bins_size, nbins);
    CSC(cudaGetLastError());

    thrust::exclusive_scan(
        thrust::device_pointer_cast(bins_size), 
        thrust::device_pointer_cast(bins_size + nbins), 
        thrust::device_pointer_cast(bins_start)
    );
    CSC(cudaGetLastError());

    // можно быть поэкономнее с размерами сетки?
    prepare_buckets<<<BINS_BLOCK_SIZE, AVG_BIN_SIZE>>>(dev_arr, buff, len, min, max, bins_start, nbins);
    CSC(cudaGetLastError());
    uint32_t *bins_end = bins_start; // bins_start becomes bins_end after prepare_buckets

    bucket_bitonic_sort<<<BINS_BLOCK_SIZE, AVG_BIN_SIZE>>>(
        buff, dev_arr, len, 
        bins_end, bins_size, 
        queue_bins_size, 
        queue_bins_start,
        nbins,
        gpu_queue_len
    );
    CSC(cudaGetLastError());
    
    uint32_t cpu_queue_len;
    uint32_t *test = (uint32_t *) malloc(sizeof(uint32_t));
    CSC(cudaMemcpy(test, gpu_queue_len, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cpu_queue_len = *test;
    free(test);

    if (cpu_queue_len) {
        rec_cnt++;
        uint32_t *queue_bins_size_cpu = (uint32_t *) malloc(sizeof(uint32_t) * cpu_queue_len);
        CSC(cudaMemcpy(queue_bins_size_cpu, queue_bins_size, sizeof(uint32_t) * cpu_queue_len, cudaMemcpyDeviceToHost));

        uint32_t *queue_bins_start_cpu = (uint32_t *) malloc(sizeof(uint32_t) * cpu_queue_len);
        CSC(cudaMemcpy(queue_bins_start_cpu, queue_bins_start, sizeof(uint32_t) * cpu_queue_len, cudaMemcpyDeviceToHost)); 

        for (uint32_t i = 0; i < cpu_queue_len; ++i) {
            recursive_bucket_sort(
                buff + queue_bins_start_cpu[i],
                dev_arr + queue_bins_start_cpu[i],
                queue_bins_size_cpu[i],
                bins_start, bins_size,
                gpu_queue_len,
                queue_bins_start,
                queue_bins_size
            );
            CSC(cudaMemcpy(
                dev_arr + queue_bins_start_cpu[i],
                buff + queue_bins_start_cpu[i],
                sizeof(float) * queue_bins_size_cpu[i],
                cudaMemcpyDeviceToDevice
            ));
        }

        free(queue_bins_size_cpu);
        free(queue_bins_start_cpu);
    }
    global_max = max;
    global_min = min;
}


void bucket_sort(float *arr, uint32_t len)
{
    if (len <= 1) {
        return;
    }

    float *dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(float) * len));
    CSC(cudaMemcpy(dev_arr, arr, sizeof(float) * len, cudaMemcpyHostToDevice));

    float *buff;
    CSC(cudaMalloc(&buff, sizeof(float) * len));


    uint32_t AVG_BIN_SIZE = BINS_SHMEM_SIZE / 2;
    uint32_t nbins = ceil(float(len) / AVG_BIN_SIZE);

    uint32_t *bins_size;
    CSC(cudaMalloc(&bins_size, sizeof(uint32_t) * nbins));

    uint32_t *bins_start;
    CSC(cudaMalloc(&bins_start, sizeof(uint32_t) * nbins));


    uint32_t *gpu_queue_len;
    CSC(cudaMalloc(&gpu_queue_len, sizeof(uint32_t)));

    uint32_t *queue_bins_size;
    CSC(cudaMalloc(&queue_bins_size, sizeof(uint32_t) * nbins / 2));

    uint32_t *queue_bins_start;
    CSC(cudaMalloc(&queue_bins_start, sizeof(uint32_t) * nbins / 2));


    recursive_bucket_sort(
        dev_arr, buff, len,
        bins_start, bins_size, 
        gpu_queue_len,
        queue_bins_start,
        queue_bins_size
    );

    CSC(cudaMemcpy(arr, dev_arr, sizeof(float) * len, cudaMemcpyDeviceToHost));


    CSC(cudaFree(queue_bins_start));
    CSC(cudaFree(queue_bins_size));
    CSC(cudaFree(gpu_queue_len));

    CSC(cudaFree(bins_start));
    CSC(cudaFree(bins_size));

    CSC(cudaFree(buff));
    CSC(cudaFree(dev_arr));
}


void init_array(float *arr, uint32_t len)
{
    std::random_device randdev;
    std::mt19937 generator(randdev());
    //std::uniform_real_distribution<double> distrib(-FLT_MAX, FLT_MAX);
    //std::normal_distribution<> distrib{0,10000000};
    //std::lognormal_distribution<> distrib(1.6, 0.25);

    int nclusters = 4;
    //float clusters[] = {500, nextafterf(-1, -FLT_MAX), nextafterf(500, FLT_MAX), -1};
    float clusters[] = {-1, nextafterf(-1, -FLT_MAX), 500, nextafterf(500, FLT_MAX)};

    for (int j = 0; j < nclusters; ++j) {
        for (uint32_t i = (len * double(j) / nclusters); i < (len * (double(j + 1) / nclusters)); ++i) { 
            arr[i] = clusters[nclusters - 1 - j];
            //arr[i] = distrib(generator);
        }
    }
    // float min = -FLT_MAX;
    // float max = FLT_MAX;
    // double step = (double(max) - min) / len;
    // for (uint32_t i = 0; i < len; ++i) {
    //     arr[i] = max - step * i;
    // }
}


int main()
{
    uint32_t len;
    float *arr;

#ifdef DEBUG
    scanf("%u", &len);
    arr = (float *) malloc(sizeof(float) * len);
    init_array(arr, len);
#else 
    freopen(NULL, "rb", stdin);
    fread(&len, sizeof(uint32_t), 1, stdin);
    arr = (float *) malloc(sizeof(float) * len);
    fread(arr, sizeof(float), len, stdin);
#endif
    
    bucket_sort(arr, len);

#ifdef DEBUG
    uint32_t i = 1;
    while (i < len && arr[i - 1] <= arr[i]) {
        ++i;
    }
    if (i >= len) {
        printf("Array has been successfully sorted!\n");
    } else {
        printf("FAILED at %u :(\n", i);
        for (uint32_t j = fmaxf(0, int(i) - 7); j < fminf(i + 10, len); ++j) {
            printf("%.8f ", arr[j]);
        }
        printf("\n");
    }
#else
    uint32_t i = 1;
    while (i < len && arr[i - 1] <= arr[i]) {
        ++i;
    }
    if (i < len) {
        fprintf(stderr, "FAILED at %u with len %u:(\n", i, len);
        fprintf(stderr, "max %e, min %e, rec %u\n", global_max, global_min, rec_cnt);
        for (uint32_t j = fmaxf(0, int(i) - 10); j < fminf(i + 10, len); ++j) {
            fprintf(stderr, "%e ", arr[j]);
        }
        fprintf(stderr, "\n");
        exit(0);
    }
    freopen(NULL, "wb", stdout);
    fwrite(arr, sizeof(float), len, stdout);
#endif

    free(arr);
    
    return 0;
}
