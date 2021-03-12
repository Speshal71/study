#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>


class HeatMap
{
private:
    int px, py, pz;
    int nx, ny, nz;
    double hx, hy, hz;

    int x, y, z;
    int n_down, n_up, n_left, n_right, n_front, n_back;

    double *u_prev;
    double *u_cur;

    double *buff_xy;
    double *buff_xz;
    double *buff_yz;

    enum AXES {X = 1, Y = 2, Z = 3};

    typedef size_t (HeatMap::*index_func_type)(int, int, int);


    int get_rank(int x, int y, int z);
    void set_limits(
        int *k_lower, int *k_upper, 
        int *j_lower, int *j_upper, 
        int *i_lower, int *i_upper,
        int dir,
        bool copy = false
    );
    void init_boundary(double val, int plane);

    inline size_t index_xy(int i, int j, int k);
    inline size_t index_xz(int i, int j, int k);
    inline size_t index_yz(int i, int j, int k);
    index_func_type get_index_func(int axis);
    void set_boundary(double *buff, int axis);

    void boundary_to_buff(double *buff, int axis);
    
    void sendrecv_along_axis(
        int axis, int pos,
        int neighbor_down, int neighbor_up,
        double *buff, size_t buff_len
    );
    void exchange_boundaries();

    inline size_t to_index(int i, int j, int k)
    {
        return k * nx * ny + j * nx + i;
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

    inline double& index(int i, int j, int k)
    {
        return u_cur[k * nx * ny + j * nx + i];
    }

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

    this->u_prev = new double[nz * ny * nx];
    for (int i = 0; i < (nz * ny * nx); ++i) {
        u_prev[i] = u_0;
    }

    if (n_left < 0)  init_boundary(u_left, -X);
    if (n_right < 0) init_boundary(u_right, X);
    if (n_front < 0) init_boundary(u_front, -Y);
    if (n_back < 0)  init_boundary(u_back, Y);
    if (n_down < 0)  init_boundary(u_down, -Z);
    if (n_up < 0)    init_boundary(u_up, Z);

    this->u_cur = new double[nz * ny * nx];
    memcpy(this->u_cur, this->u_prev, sizeof(double) * nz * ny * nx);

    buff_xy = new double[nx * ny];
    buff_xz = new double[nx * nz];
    buff_yz = new double[ny * nz];
}


HeatMap::~HeatMap()
{
    delete[] u_prev;
    delete[] u_cur;
    delete[] buff_xy;
    delete[] buff_xz;
    delete[] buff_yz;
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

    for (int k = k_lower; k < k_upper; ++k) {
        for (int j = j_lower; j < j_upper; ++j) {
            for (int i = i_lower; i < i_upper; ++i) {
                u_prev[k * nx * ny + j * nx + i] = val;
            }
        }
    }
}


inline size_t HeatMap::index_xy(int i, int j, int k)
{
    return j * nx + i;
}


inline size_t HeatMap::index_xz(int i, int j, int k)
{
    return k * nx + i;
}


inline size_t HeatMap::index_yz(int i, int j, int k)
{
    return k * ny + j;
}


HeatMap::index_func_type HeatMap::get_index_func(int axis)
{
    if (axis == Z || axis == -Z) {
        return &HeatMap::index_xy;
    }
    else if (axis == Y || axis == -Y) {
        return &HeatMap::index_xz;
    }
    else {
        return &HeatMap::index_yz;
    }
}


void HeatMap::boundary_to_buff(double *buff, int axis)
{
    int k_lower, k_upper, j_lower, j_upper, i_lower, i_upper;
    set_limits(&k_lower, &k_upper, &j_lower, &j_upper, &i_lower, &i_upper, axis, true);

    index_func_type index_func = get_index_func(axis);

    for (int k = k_lower; k < k_upper; ++k) {
        for (int j = j_lower; j < j_upper; ++j) {
            for (int i = i_lower; i < i_upper; ++i) {
                buff[(this->*index_func)(i, j, k)] =  u_prev[k * nx * ny + j * nx + i];
            }
        }
    }
}


void HeatMap::set_boundary(double *buff, int axis)
{
    int k_lower, k_upper, j_lower, j_upper, i_lower, i_upper;
    set_limits(&k_lower, &k_upper, &j_lower, &j_upper, &i_lower, &i_upper, axis);

    index_func_type index_func = get_index_func(axis);

    for (int k = k_lower; k < k_upper; ++k) {
        for (int j = j_lower; j < j_upper; ++j) {
            for (int i = i_lower; i < i_upper; ++i) {
                u_prev[k * nx * ny + j * nx + i] = buff[(this->*index_func)(i, j, k)];
            }
        }
    }
}


void HeatMap::sendrecv_along_axis(int axis, int axis_pos,
                                  int neighbor_down, int neighbor_up,
                                  double *buff, size_t buff_len)
{
    MPI_Status status;

    if (axis_pos % 2) {
        if (neighbor_up >= 0) {
            boundary_to_buff(buff, axis);
            MPI_Send(buff, buff_len, MPI_DOUBLE, neighbor_up, 0, MPI_COMM_WORLD);
        }
        if (neighbor_down >= 0) {
            MPI_Recv(buff, buff_len, MPI_DOUBLE, neighbor_down, 0, MPI_COMM_WORLD, &status);
            set_boundary(buff, -axis);
        }
    } 
    else {
        if (neighbor_down >= 0) {
            MPI_Recv(buff, buff_len, MPI_DOUBLE, neighbor_down, 0, MPI_COMM_WORLD, &status);
            set_boundary(buff, -axis);
        }
        if (neighbor_up >= 0) {
            boundary_to_buff(buff, axis);
            MPI_Send(buff, buff_len, MPI_DOUBLE, neighbor_up, 0, MPI_COMM_WORLD);
        }
    }
}


void HeatMap::exchange_boundaries()
{
    sendrecv_along_axis(X, x, n_left, n_right, buff_yz, ny * nz);
    sendrecv_along_axis(-X, x, n_right, n_left, buff_yz, ny * nz);
    
    sendrecv_along_axis(Y, y, n_front, n_back, buff_xz, nx * nz);
    sendrecv_along_axis(-Y, y, n_back, n_front, buff_xz, nx * nz);

    sendrecv_along_axis(Z, z, n_down, n_up, buff_xy, nx * ny);
    sendrecv_along_axis(-Z, z, n_up, n_down, buff_xy, nx * ny);
}


inline double max(double a, double b)
{
    return (a > b ? a : b);
}


double HeatMap::approximate()
{
    double *temp_map = u_prev;
    u_prev = u_cur;
    u_cur = temp_map;

    exchange_boundaries();

    double local_eps = -1;

    for (int k = 1; k < (nz - 1); ++k) {
        for (int j = 1; j < (ny - 1); ++j) {
            for (int i = 1; i < (nx - 1); ++i) {
                u_cur[to_index(i, j, k)] = (
                    (u_prev[to_index(i + 1, j, k)] + u_prev[to_index(i - 1, j, k)]) / (hx * hx) +
                    (u_prev[to_index(i, j + 1, k)] + u_prev[to_index(i, j - 1, k)]) / (hy * hy) +
                    (u_prev[to_index(i, j, k + 1)] + u_prev[to_index(i, j, k - 1)]) / (hz * hz)
                ) / (
                    2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz)
                );

                local_eps = max(local_eps, std::abs(u_cur[to_index(i, j, k)] - u_prev[to_index(i, j, k)]));
            }
        }
    }

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
        file = fopen(filepath, "w");
    }

    for (int kp = 0; kp < pz; ++kp) {
        for (int k = 1; k < (nz - 1); ++k) {
            for (int jp = 0; jp < py; ++jp) {
                for (int j = 1; j < (ny - 1); ++j) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (my_rank == 0) {
                        for (int ip = 0; ip < px; ++ip) {
                            int send_to = get_rank(ip, jp, kp);

                            if (send_to == 0) {
                                memcpy(buff_xy, &(this->u_cur[to_index(0, j, k)]), sizeof(double) * nx);
                            } 
                            else {
                                MPI_Recv(buff_xy, nx, MPI_DOUBLE, get_rank(ip, jp, kp), 0, MPI_COMM_WORLD, &status);
                            }

                            for (int i = 1; i < (nx - 1); ++i) {
                                fprintf(file, "%.6e ", buff_xy[i]);
                            }
                        }

                    }
                    else if (z == kp && y == jp) {
                        MPI_Send(&(this->u_cur[to_index(0, j, k)]), nx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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

    heat_map.write(filepath);

    MPI_Finalize();

    delete[] filepath;

    return 0;
}