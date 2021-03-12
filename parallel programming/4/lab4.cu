#include <stdio.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define HALFWARP 16

#define CSC(call)  					                                \
do {								                                \
	cudaError_t res = call;		                                	\
	if (res != cudaSuccess) {	                                	\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);				                                   	\
	}							                                   	\
} while(0)


__global__ void swap(double *A, int n, size_t pitch, int i, int j)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

	while (idx < (n + 1)) {
        double temp = A[idx * pitch + i];
        A[idx * pitch + i] = A[idx * pitch + j];
        A[idx * pitch + j] = temp;
		idx += offset;
	}
}


__device__ int align_row(int i)
{
    return (i / HALFWARP) * HALFWARP;
}


__global__ void rows_elimination(double *A, int n, size_t pitch, int left_upper_corner)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;

    int start_row = left_upper_corner + 1;
    double pivot = A[left_upper_corner * pitch + left_upper_corner];
    double top_elem;

    for (int j = left_upper_corner + 1 + idy; j < (n + 1); j += offset_y) {
        top_elem = A[j * pitch + left_upper_corner];
        for (int i = align_row(start_row) + idx; i < n; i += offset_x) {
            if (i >= start_row) {
                A[j * pitch + i] -= A[left_upper_corner * pitch + i] / pivot * top_elem;
            }
        }
    }
}


__global__ void backward_elimination(double *A, int n, size_t pitch, int i)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    double coef = A[n * pitch + i] / A[i * pitch + i];

	while (idx < i) {
        A[n * pitch + idx] -= coef * A[i * pitch + idx];
		idx += offset;
    }
}


__global__ void normalize(double *A, int n, size_t pitch)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

	while (idx < n) {
        A[n * pitch + idx] /= A[idx * pitch + idx];
		idx += offset;
    }
}


struct comparator {
	__host__ __device__ bool operator()(double a, double b) {
		return abs(a) < abs(b); 
	}
};


void solve(double *A, double *b, double *x, int n)
{
    // solves linear system Ax = b
    // argument A must be transposed matrix A

    double *A_dev; // we will store both A and b in A_dev variable
    size_t pitch_size;
    CSC(cudaMallocPitch(&A_dev, &pitch_size, sizeof(double) * n, n + 1));
    size_t pitch = pitch_size / sizeof(double);
    CSC(cudaMemcpy2D(
        A_dev, pitch_size, 
        A, sizeof(double) * n,  
        sizeof(double) * n, n, 
        cudaMemcpyHostToDevice
    ));
    CSC(cudaMemcpy(A_dev + pitch * n, b, sizeof(double) * n, cudaMemcpyHostToDevice));

    comparator comp;

    // transform A to upper triangular matrix
    for (int i = 0; i < n; ++i) {
        // search for the pivot value
        thrust::device_ptr<double> col_p = thrust::device_pointer_cast(&A_dev[i * pitch]);
        thrust::device_ptr<double> max_p = thrust::max_element(col_p + i, col_p + n, comp);
        int j = max_p - col_p; // the row to be swapped with i'th row
        
        if (j > i) {
            swap<<<1, 64>>>(A_dev, n, pitch, i, j);
        }

        rows_elimination<<<dim3(2, 2), dim3(32, 32)>>>(A_dev, n, pitch, i);
    }
    
    // transform A to identity matrix
    for (int i = n - 1; i >= 0; --i) {
        backward_elimination<<<1, 64>>>(A_dev, n, pitch, i);
    }

    normalize<<<1, 64>>>(A_dev, n, pitch);
    
    CSC(cudaMemcpy(x, A_dev + pitch * n, sizeof(double) * n, cudaMemcpyDeviceToHost));

    CSC(cudaFree(A_dev));
}


int main()
{
    int n;
    double *A;
    double *b;
    double *x;

    scanf("%d", &n);

    A = (double *) malloc(sizeof(double) * n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%lf", &A[j * n + i]);
        }
    }

    b = (double *) malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) {
        scanf("%lf", &b[i]);
    }

    x = (double *) malloc(sizeof(double) * n);

    // matrix A should be transposed
    solve(A, b, x, n);
    
    for (int i = 0; i < n; ++i) {
        printf("%.10e ", x[i]);
    }
    printf("\n");
    

    free(A);
    free(b);
    free(x);
    
	return 0;
}