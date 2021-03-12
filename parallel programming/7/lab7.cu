#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/tuple.h>
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

#define X 1
#define Y 2
#define Z 3


__global__ void init_array(double *arr, double val, size_t len)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for(size_t i = tid; i < len; i += offset) {
        arr[i] = val;
    }
}


__device__ inline size_t to_index_gpu(size_t pitchx, int ny, int nz, int i, int j, int k)
{
    return k * pitchx * ny + j * pitchx + i;
}


__global__ void print_array(double *arr, size_t pitchx, int nx, int ny, int nz)
{
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                printf("%lf ", arr[to_index_gpu(pitchx, ny, nz, i, j, k)]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


__device__ inline size_t index_xy(int nx, int ny, int nz, int i, int j, int k)
{
    return j * nx + i;
}


__device__ inline size_t index_xz(int nx, int ny, int nz, int i, int j, int k)
{
    return k * nx + i;
}


__device__ inline size_t index_yz(int nx, int ny, int nz, int i, int j, int k)
{
    return k * ny + j;
}


__device__ size_t (*get_index_func(int axis))(int, int, int, int, int, int)
{
    if (axis == Z || axis == -Z) {
        return &index_xy;
    }
    else if (axis == Y || axis == -Y) {
        return &index_xz;
    }
    else {
        return &index_yz;
    }
}


__global__ void init_boundary_kernel(double *arr, double val, 
                                     size_t pitchx, int nx, int ny, int nz,
                                     int k_lower, int k_upper,  
                                     int j_lower, int j_upper, 
                                     int i_lower, int i_upper)
{
    for (int k = k_lower; k < k_upper; ++k) {
        for (int j = j_lower; j < j_upper; ++j) {
            for (int i = i_lower; i < i_upper; ++i) {
                arr[to_index_gpu(pitchx, ny, nz, i, j, k)] = val;
            }
        }
    }
}


__global__ void boundary_to_buff_kernel(double *arr, double *buff, int axis,
                                        size_t pitchx, int nx, int ny, int nz,
                                        int k_lower, int k_upper,  
                                        int j_lower, int j_upper, 
                                        int i_lower, int i_upper)
{
    size_t (*index_func)(int, int, int, int, int, int) = get_index_func(axis);

    for (int k = k_lower; k < k_upper; ++k) {
        for (int j = j_lower; j < j_upper; ++j) {
            for (int i = i_lower; i < i_upper; ++i) {
                buff[(*index_func)(nx, ny, nz, i, j, k)] =  arr[to_index_gpu(pitchx, ny, nz, i, j, k)];
            }
        }
    }
}


__global__ void set_boundary_kernel(double *arr_prev, double *arr_cur,
                                    double *buff, int axis,
                                    size_t pitchx, int nx, int ny, int nz,
                                    int k_lower, int k_upper,  
                                    int j_lower, int j_upper, 
                                    int i_lower, int i_upper)
{
    size_t (*index_func)(int, int, int, int, int, int) = get_index_func(axis);

    double elem;
    for (int k = k_lower; k < k_upper; ++k) {
        for (int j = j_lower; j < j_upper; ++j) {
            for (int i = i_lower; i < i_upper; ++i) {
                elem = buff[(*index_func)(nx, ny, nz, i, j, k)];
                arr_prev[to_index_gpu(pitchx, ny, nz, i, j, k)] = elem;
                arr_cur[to_index_gpu(pitchx, ny, nz, i, j, k)] = elem;
            }
        }
    }
}


class HeatMap
{
private:
    int px, py, pz;
    int nx, ny, nz;
    double hx, hy, hz;

    size_t pitchx;

    int x, y, z;
    int n_down, n_up, n_left, n_right, n_front, n_back;

    double *u_prev;
    double *u_cur;

    double *buff_xy_cpu, *buff_xy_gpu;
    double *buff_xz_cpu, *buff_xz_gpu;
    double *buff_yz_cpu, *buff_yz_gpu;

    void set_gpu(int my_rank);
    int get_rank(int x, int y, int z);
    void set_limits(
        int *k_lower, int *k_upper, 
        int *j_lower, int *j_upper, 
        int *i_lower, int *i_upper,
        int dir,
        bool copy = false
    );

    void init_boundary(double val, int plane);
    void set_boundary(double *buff, double *buff_dev, size_t buff_len, int axis);
    void boundary_to_buff(double *buff, double *buff_dev, size_t buff_len, int axis);
    
    void sendrecv_along_axis(
        int axis, int pos,
        int neighbor_down, int neighbor_up,
        double *buff, double *buff_dev, size_t buff_len
    );
    void exchange_boundaries();

    inline size_t to_index(int i, int j, int k)
    {
        return k * pitchx * ny + j * pitchx + i;
    }


public:
    HeatMap(
        int px, int py, int pz, 
        int nx, int ny, int nz,
        double lx, double ly, double lz,
        double u_down, double u_up, double u_left, 
        double u_right, double u_front, double u_back,
        double u_0,
        int my_rank
    );

    double approximate();
    void write(char *filepath);

    ~HeatMap();
};


HeatMap::HeatMap(int px, int py, int pz, 
                 int nx, int ny, int nz,
                 double lx, double ly, double lz,
                 double u_down, double u_up, double u_left, 
                 double u_right, double u_front, double u_back,
                 double u_0,
                 int my_rank)
{   
    nx += 2;
    ny += 2;
    nz += 2;

    this->px = px;
    this->py = py;
    this->pz = pz;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->hx = lx / (this->nx - 2) / px;
    this->hy = ly / (this->ny - 2) / py;
    this->hz = lz / (this->nz - 2) / pz;

    this->z = my_rank / (px * py);
    this->y = (my_rank % (px * py)) / px;
    this->x = (my_rank % (px * py)) % px;

    this->n_left  = get_rank(x - 1, y, z);
    this->n_right = get_rank(x + 1, y, z);
    this->n_front = get_rank(x, y - 1, z); 
    this->n_back  = get_rank(x, y + 1, z);
    this->n_down  = get_rank(x, y, z - 1);
    this->n_up    = get_rank(x, y, z + 1);

    set_gpu(my_rank);

    CSC(cudaMallocPitch(&this->u_prev, &this->pitchx, sizeof(double) * nx, ny * nz));
    CSC(cudaMallocPitch(&this->u_cur, &this->pitchx, sizeof(double) * nx, ny * nz));
    this->pitchx = this->pitchx / sizeof(double);
    
    init_array<<<16, 16>>>(this->u_prev, u_0, pitchx * ny * nz);
    CSC(cudaGetLastError());

    // print_array<<<1, 1>>>(this->u_prev, pitchx, nx, ny, nz);
    // fflush(stdout);

    if (n_left < 0)  init_boundary(u_left, -X);
    if (n_right < 0) init_boundary(u_right, X);
    if (n_front < 0) init_boundary(u_front, -Y);
    if (n_back < 0)  init_boundary(u_back, Y);
    if (n_down < 0)  init_boundary(u_down, -Z);
    if (n_up < 0)    init_boundary(u_up, Z);

    // print_array<<<1, 1>>>(this->u_prev, pitchx, nx, ny, nz);
    // fflush(stdout);

    CSC(cudaMemcpy(this->u_cur, this->u_prev, sizeof(double) * nz * ny * pitchx, cudaMemcpyDeviceToDevice));

    CSC(cudaMalloc(&buff_xy_gpu, sizeof(double) * nx * ny));
    CSC(cudaMalloc(&buff_xz_gpu, sizeof(double) * nx * nz));
    CSC(cudaMalloc(&buff_yz_gpu, sizeof(double) * ny * nz));

    buff_xy_cpu = new double[nx * ny];
    buff_xz_cpu = new double[nx * nz];
    buff_yz_cpu = new double[ny * nz];

    // print_array<<<1, 1>>>(this->u_prev, pitchx, nx, ny, nz);
    // print_array<<<1, 1>>>(this->u_cur, pitchx, nx, ny, nz);
    // fflush(stdout);

}


HeatMap::~HeatMap()
{
    CSC(cudaFree(u_prev));
    CSC(cudaFree(u_cur));

    CSC(cudaFree(buff_xy_gpu));
    CSC(cudaFree(buff_xz_gpu));
    CSC(cudaFree(buff_yz_gpu));

    delete[] buff_xy_cpu;
    delete[] buff_xz_cpu;
    delete[] buff_yz_cpu;
}



void HeatMap::set_gpu(int my_rank)
{
    int devicesCount;
    MPI_Comm local_comm;
    int local_rank;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank,  MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    
    CSC(cudaGetDeviceCount(&devicesCount));
    CSC(cudaSetDevice(local_rank % devicesCount));
}


int HeatMap::get_rank(int x, int y, int z)
{
    int rank;

    if (x < 0 || x >= px || y < 0 || y >= py || z < 0 || z >= pz) {
        rank = -1;
    } 
    else {
        rank = z * px * py + y * px + x;
    }

    return rank;
}


void HeatMap::set_limits(int *k_lower, int *k_upper, 
                         int *j_lower, int *j_upper, 
                         int *i_lower, int *i_upper,
                         int dir,
                         bool copy)
{
    *k_lower = 0, *k_upper = nz;
    *j_lower = 0, *j_upper = ny;
    *i_lower = 0, *i_upper = nx;

    int copy_shift = (copy ? 1 : 0);

    if (dir == -Z) {
        *k_lower = 0 + copy_shift;
        *k_upper = 1 + copy_shift;
    } 
    else if (dir == Z) {
        *k_lower = nz - 1 - copy_shift;
        *k_upper = nz - copy_shift;
    }
    else if (dir == -Y) {
        *j_lower = 0 + copy_shift;
        *j_upper = 1 + copy_shift;
    }
    else if (dir == Y) {
        *j_lower = ny - 1 - copy_shift;
        *j_upper = ny - copy_shift;
    }
    else if (dir == -X) {
        *i_lower = 0 + copy_shift;
        *i_upper = 1 + copy_shift;
    }
    else {
        *i_lower = nx - 1 - copy_shift;
        *i_upper = nx - copy_shift;
    }
}


void HeatMap::init_boundary(double val, int dir)
{
    int k_lower, k_upper, j_lower, j_upper, i_lower, i_upper;
    set_limits(&k_lower, &k_upper, &j_lower, &j_upper, &i_lower, &i_upper, dir);

    init_boundary_kernel<<<1, 1>>>(
        u_prev, val,
        pitchx, nx, ny, nz,
        k_lower, k_upper,
        j_lower, j_upper,
        i_lower, i_upper
    );
    CSC(cudaGetLastError());
}


void HeatMap::boundary_to_buff(double *buff, double *buff_dev, size_t buff_len, int axis)
{
    int k_lower, k_upper, j_lower, j_upper, i_lower, i_upper;
    set_limits(&k_lower, &k_upper, &j_lower, &j_upper, &i_lower, &i_upper, axis, true);

    boundary_to_buff_kernel<<<1, 1>>>(
        u_prev, buff_dev, axis,
        pitchx, nx, ny, nz,
        k_lower, k_upper,
        j_lower, j_upper,
        i_lower, i_upper
    );
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(buff, buff_dev, buff_len * sizeof(double), cudaMemcpyDeviceToHost));
}


void HeatMap::set_boundary(double *buff, double *buff_dev, size_t buff_len, int axis)
{
    int k_lower, k_upper, j_lower, j_upper, i_lower, i_upper;
    set_limits(&k_lower, &k_upper, &j_lower, &j_upper, &i_lower, &i_upper, axis);

    CSC(cudaMemcpy(buff_dev, buff, buff_len * sizeof(double), cudaMemcpyHostToDevice));

    set_boundary_kernel<<<1, 1>>>(
        u_prev, u_cur, 
        buff_dev, axis,
        pitchx, nx, ny, nz,
        k_lower, k_upper,
        j_lower, j_upper,
        i_lower, i_upper
    );
    CSC(cudaGetLastError());
}


void HeatMap::sendrecv_along_axis(int axis, int axis_pos,
                                  int neighbor_down, int neighbor_up,
                                  double *buff, double *buff_dev, size_t buff_len)
{
    MPI_Status status;

    if (axis_pos % 2) {
        if (neighbor_up >= 0) {
            boundary_to_buff(buff, buff_dev, buff_len, axis);
            MPI_Send(buff, buff_len, MPI_DOUBLE, neighbor_up, 0, MPI_COMM_WORLD);
        }
        if (neighbor_down >= 0) {
            MPI_Recv(buff, buff_len, MPI_DOUBLE, neighbor_down, 0, MPI_COMM_WORLD, &status);
            set_boundary(buff, buff_dev, buff_len, -axis);
        }
    } 
    else {
        if (neighbor_down >= 0) {
            MPI_Recv(buff, buff_len, MPI_DOUBLE, neighbor_down, 0, MPI_COMM_WORLD, &status);
            set_boundary(buff, buff_dev, buff_len, -axis);
        }
        if (neighbor_up >= 0) {
            boundary_to_buff(buff, buff_dev, buff_len, axis);
            MPI_Send(buff, buff_len, MPI_DOUBLE, neighbor_up, 0, MPI_COMM_WORLD);
        }
    }
}


void HeatMap::exchange_boundaries()
{
    sendrecv_along_axis(X, x, n_left, n_right, buff_yz_cpu, buff_yz_gpu, ny * nz);
    sendrecv_along_axis(-X, x, n_right, n_left, buff_yz_cpu, buff_yz_gpu, ny * nz);
    
    sendrecv_along_axis(Y, y, n_front, n_back, buff_xz_cpu, buff_xz_gpu, nx * nz);
    sendrecv_along_axis(-Y, y, n_back, n_front, buff_xz_cpu, buff_xz_gpu, nx * nz);

    sendrecv_along_axis(Z, z, n_down, n_up, buff_xy_cpu, buff_xy_gpu, nx * ny);
    sendrecv_along_axis(-Z, z, n_up, n_down, buff_xy_cpu, buff_xy_gpu, nx * ny);
}


__global__ void approximate_kernel(double *u_prev, double *u_cur, 
                                   size_t pitchx, int nx, int ny, int nz,
                                   double hx, double hy, double hz)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;
    int offset_z = gridDim.z * blockDim.z;

    for (int k = 1 + idz; k < (nz - 1); k += offset_z) {
        for (int j = 1 + idy; j < (ny - 1); j += offset_y) {
            for (int i = idx; i < (nx - 1); i += offset_x) {
                if (i > 0) {
                    u_cur[to_index_gpu(pitchx, ny, nz, i, j, k)] = (
                        (u_prev[to_index_gpu(pitchx, ny, nz, i + 1, j, k)] + u_prev[to_index_gpu(pitchx, ny, nz, i - 1, j, k)]) / (hx * hx) +
                        (u_prev[to_index_gpu(pitchx, ny, nz, i, j + 1, k)] + u_prev[to_index_gpu(pitchx, ny, nz, i, j - 1, k)]) / (hy * hy) +
                        (u_prev[to_index_gpu(pitchx, ny, nz, i, j, k + 1)] + u_prev[to_index_gpu(pitchx, ny, nz, i, j, k - 1)]) / (hz * hz)
                    ) / (
                        2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz)
                    );
                }
            }
        }
    }
}


struct MaxDiffComparator 
{
    __device__ bool operator()(thrust::tuple<double, double> reduced_pair, 
                               thrust::tuple<double, double> pair) 
    {
        double reduced_max = fabs(thrust::get<1>(reduced_pair) - thrust::get<0>(reduced_pair));
        double pair_diff = fabs(thrust::get<1>(pair) - thrust::get<0>(pair));
        
        return reduced_max < pair_diff;
    }
};


double HeatMap::approximate()
{
    double *temp_map = u_prev;
    u_prev = u_cur;
    u_cur = temp_map;

    exchange_boundaries();

    approximate_kernel<<<dim3(1, 1, 4), dim3(32, 2, 1)>>>(
        u_prev, u_cur, 
        pitchx, nx, ny, nz,
        hx, hy, hz
    );
    CSC(cudaGetLastError());

    thrust::device_ptr<double> thrust_u_prev = thrust::device_pointer_cast(u_prev);
    thrust::device_ptr<double> thrust_u_cur = thrust::device_pointer_cast(u_cur);
    auto pair_grid = thrust::make_zip_iterator(thrust::make_tuple(thrust_u_cur, thrust_u_prev));

    MaxDiffComparator max_diff_comparator;
    thrust::tuple<double, double> max_diff = *thrust::max_element(pair_grid, pair_grid + (pitchx * ny * nz), max_diff_comparator);
    double local_eps = std::fabs(thrust::get<1>(max_diff) - thrust::get<0>(max_diff));

    double global_eps;
    MPI_Allreduce(&local_eps, &global_eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return global_eps;
}


void HeatMap::write(char *filepath)
{
    MPI_Status status;
    FILE *file;

    int my_rank = get_rank(x, y, z);

    if (my_rank == 0) {
        CSC(cudaMemcpy(buff_xy_cpu, this->u_cur + to_index(0, 1, 1), sizeof(double) * nx, cudaMemcpyDeviceToHost));
        for (int i = 1; i < (nx - 1); ++i) {
            fprintf(stderr, "%.6e ", buff_xy_cpu[i]);
        }
        fprintf(stderr, "\n");

        file = fopen(filepath, "w");
    }

    for (int kp = 0; kp < pz; ++kp) {
        for (int k = 1; k < (nz - 1); ++k) {
            for (int jp = 0; jp < py; ++jp) {
                for (int j = 1; j < (ny - 1); ++j) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (my_rank == 0) {
                        for (int ip = 0; ip < px; ++ip) {
                            int recv_from = get_rank(ip, jp, kp);

                            if (recv_from == 0) {
                                CSC(cudaMemcpy(buff_xy_cpu, this->u_cur + to_index(0, j, k), sizeof(double) * nx, cudaMemcpyDeviceToHost));
                            } 
                            else {
                                MPI_Recv(buff_xy_cpu, nx, MPI_DOUBLE, recv_from, 0, MPI_COMM_WORLD, &status);
                            }

                            for (int i = 1; i < (nx - 1); ++i) {
                                fprintf(file, "%.6e ", buff_xy_cpu[i]);
                            }
                        }
                        if (my_rank == 0) {
                            fprintf(file, "\n");
                        }

                    }
                    else if (z == kp && y == jp) {
                        CSC(cudaMemcpy(buff_xy_cpu, this->u_cur + to_index(0, j, k), sizeof(double) * nx, cudaMemcpyDeviceToHost));
                        MPI_Send(buff_xy_cpu, nx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                }
            }
            if (my_rank == 0) {
                fprintf(file, "\n");
            }
        }
    }

    if (my_rank == 0) {
        fclose(file);
    }
}


int main(int argc, char** argv) {
    int rank, size;
    int px, py, pz;
    int nx, ny, nz;
    char *filepath = nullptr;
    double eps;
    double lx, ly, lz;
    double u_down, u_up, u_left, u_right, u_front, u_back;
    double u_0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Enter data and broadcast to other processes
    if (rank == 0) {
        scanf("%d %d %d", &px, &py, &pz);
        scanf("%d %d %d", &nx, &ny, &nz);
        filepath = new char[100];
        scanf("%s", filepath);
        scanf("%lf", &eps);
        scanf("%lf %lf %lf", &lx, &ly, &lz);
        scanf("%lf %lf %lf %lf %lf %lf", &u_down, &u_up, &u_left, &u_right, &u_front, &u_back);
        scanf("%lf", &u_0);

        fprintf(stderr, "%d %d %d\n", px, py, pz);
        fprintf(stderr, "%d %d %d\n", nx, ny, nz);
        fprintf(stderr, "%lf\n", eps);
        fprintf(stderr, "%lf %lf %lf\n", lx, ly, lz);
        fprintf(stderr, "%lf %lf %lf %lf %lf %lf\n", u_down, u_up, u_left, u_right, u_front, u_back);
        fprintf(stderr, "%lf\n", u_0);
    }
    
    MPI_Bcast(&px, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&py, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&u_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    HeatMap heat_map(
        px, py, pz, 
        nx, ny, nz,
        lx, ly, lz,
        u_down, u_up, u_left, 
        u_right, u_front, u_back,
        u_0,
        rank
    );

    double global_eps = eps + 1;

    while (global_eps > eps) {
        global_eps = heat_map.approximate();
    }

    if (rank == 0) {
        fprintf(stderr, "eps = %e\n", global_eps);
    }

    heat_map.write(filepath);

    MPI_Finalize();

    delete[] filepath;

    return 0;
}