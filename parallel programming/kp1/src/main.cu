#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <stdio.h>
#include <chrono>

#include "objects.cuh"
#include "ray_tracer.cuh"

#define M_PI 3.14159265358979323846
#define N_OBJECTS 3


double r_c, z_c, phi_c, A_r_c, A_z_c;
double w_r_c, w_z_c, w_phi_c, p_r_c, p_z_c;
double r_n, z_n, phi_n, A_r_n, A_z_n;
double w_r_n, w_z_n, w_phi_n, p_r_n, p_z_n;
size_t nframes;


struct PlatonicSolidParams
{
    float3 cnt;
    float3 clr;
    float rds;
    float rfl;
    float rfr;
    size_t nlights;
};


struct RectangleParams
{
    float3 a, b, c, d;
    float3 clr;
    float rfl;
    uchar4 *tex;
    int w_tex, h_tex;
};


__host__ __device__
void generate_edge_lights(PlatonicSolid *fig, size_t lights_per_edge,
                          HittableObject ***wrld, size_t *wrld_size)
{
    size_t new_size = *wrld_size + lights_per_edge * fig->n_edges();
    HittableObject **new_wrld = new HittableObject*[new_size];
    for (size_t i = 0; i < *wrld_size; ++i) {
        new_wrld[i] = (*wrld)[i];
    }
    delete *wrld;

    float3 *vertices = fig->vertices();
    int2 *edges = fig->edges();

    for (size_t i = 0; i < fig->n_edges(); ++i) {
        float3 orig = vertices[edges[i].x];
        float3 dir = vertices[edges[i].y] - orig;
        for (size_t j = 1; j < (lights_per_edge + 1); ++j) {
            new_wrld[*wrld_size + i * lights_per_edge + j - 1] = (
                new Sphere(
                    orig + j * dir / (lights_per_edge + 1), 
                    0.05,
                    make_float3(1, 1, 1),
                    1, 1,
                    true
                )
            );
        }
    }

    *wrld = new_wrld;
    *wrld_size = new_size;

    delete vertices;
    delete edges;
}


std::istream& operator>>(std::istream &in, float3 &vec)
{
    in >> vec.x >> vec.y >> vec.z;
    return in;
}


__global__ void init_world(HittableObject ***wrld, size_t *wrld_size, 
                           PlatonicSolidParams *prm, RectangleParams *rec)
{
    *wrld_size = N_OBJECTS + 1;
    *wrld = new HittableObject*[*wrld_size];
    (*wrld)[0] = new Cube(prm[0].rds, prm[0].cnt, prm[0].clr, prm[0].rfl, prm[0].rfr);
    (*wrld)[1] = new Octahedron(prm[1].rds, prm[1].cnt, prm[1].clr, prm[1].rfl, prm[1].rfr);
    (*wrld)[2] = new Dodecahedron(prm[2].rds, prm[2].cnt, prm[2].clr, prm[2].rfl, prm[2].rfr);

    //floor
    (*wrld)[3] = new Rectangle(
        rec->a, rec->b, rec->c, rec->d,
        rec->clr, rec->rfl, 0,
        rec->tex, rec->w_tex, rec->h_tex
    );

    generate_edge_lights((PlatonicSolid *) (*wrld)[0], prm[0].nlights, wrld, wrld_size);
    generate_edge_lights((PlatonicSolid *) (*wrld)[1], prm[1].nlights, wrld, wrld_size);
    generate_edge_lights((PlatonicSolid *) (*wrld)[2], prm[2].nlights, wrld, wrld_size);
}


__global__ void free_world(HittableObject **wrld, size_t wrld_size)
{
    for (size_t i = 0; i < wrld_size; ++i) {
        delete wrld[i];
    }
}


template<class RendererClass>
void render_scene(RendererClass &renderer,
                  HittableObject **wrld, size_t wrld_size,
                  float3 *lights, size_t lights_size,
                  std::string &path_out)
{
    int W = renderer.width();
    int H = renderer.height();

    uchar4 *image = (uchar4 *) malloc(sizeof(uchar4) * W * H);
    
    for (int i = 0; i < nframes; ++i) {
        std::cout << i << "\'th frame";

        float t = 2 * M_PI * i / nframes;

        float ro = r_c + A_r_c * sin(w_r_c * t + p_r_c);
        float phi = phi_c + w_phi_c * t;
        float z = z_c + A_z_c * sin(w_z_c * t + p_z_c);

        float3 cam_pos = make_float3(ro * cos(phi), ro * sin(phi), z);
        
        ro = r_n + A_r_n * sin(w_r_n * t + p_r_n);
        phi = phi_n + w_phi_n * t;
        z =  z_n + A_z_n * sin(w_z_n * t + p_z_n);

        float3 cam_look_at = make_float3(ro * cos(phi), ro * sin(phi), z);

        renderer.setUpCam(cam_pos, cam_look_at);

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        renderer.render(
            wrld, wrld_size,
            lights, lights_size,
            image
        );
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        
        std::cout << " --- " << 1000000.0 / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " fps\n";

        char picname[200];
        sprintf(picname, path_out.data(), i);

        std::ofstream fout(picname, std::ofstream::binary);

        if (!fout.is_open()) {
            std::cout << "Cannot open " << picname << "\n";
            return;
        }

        fout.write((char *) &W, sizeof(int));
        fout.write((char *) &H, sizeof(int));

        fout.write((char *) image, sizeof(uchar4) * W * H);
        fout.close();
    }  

    free(image);
}


bool parse_args(int argc, char **argv)
{
    if (argc == 1) 
    {
        return false;
    } 
    else if (argc > 2) 
    {
        std::cout << "Wrong count of arguments\n";
        exit(0);
    }
    else if (std::string(argv[1]) == "--default")  
    {
        std::cout << "40\n";
        std::cout << "./out/%d.data\n";
        std::cout << "800 400 90\n";
        std::cout << "0 -2 0 5 5 1 1 1 0 0\n";
        std::cout << "0 -5 0 1 1 0 0 1 0 0\n";
        std::cout << "1.5 -1 -3    0.2 1 0.2   2    0.7 0.5   5\n";
        std::cout << "0 0 -7       1 0 0       2    0.4 0.6   4\n";
        std::cout << "-4 2 -5      1 1 1       2    0.5 0.5   5\n";
        std::cout << "7 -3 3   -7 -3 3    -7 -3 -15   7 -3 -15   ./floor.data 1 1 1  0.3\n";
        std::cout << "2\n";
        std::cout << "0 6.5 0    1 1 1\n";
        std::cout << "0 6.5 -8   1 1 1\n";
        std::cout << "5 1\n";
        exit(0);
    } 
    else if (std::string(argv[1]) == "--cpu")
    {
        return true;
    }
    else if (std::string(argv[1]) == "--gpu")
    {
        return false;
    }
    else 
    {
        std::cout << "Wrong argument\n";
        exit(0);
    }
}

int main(int argc, char *argv[])
{
    bool is_cpu = parse_args(argc, argv);
    
    std::cin >> nframes;

    std::string out_path;
    std::cin >> out_path;

    // Camera resolution and its movement parameters
    size_t W, H, fov;
    std::cin >> W >> H >> fov;

    std::cin >> r_c >> z_c >> phi_c >> A_r_c >> A_z_c;
    std::cin >> w_r_c >> w_z_c >> w_phi_c >> p_r_c >> p_z_c;
    std::cin >> r_n >> z_n >> phi_n >> A_r_n >> A_z_n;
    std::cin >> w_r_n >> w_z_n >> w_phi_n >> p_r_n >> p_z_n;
    
    // Platonic solids data
    PlatonicSolidParams prm[N_OBJECTS];

    for (int i = 0; i < N_OBJECTS; ++i) {
        std::cin >> prm[i].cnt >> prm[i].clr >> prm[i].rds;
        std::cin >> prm[i].rfl >> prm[i].rfr >> prm[i].nlights;
    }

    // Floor data
    float3 flr_a, flr_b, flr_c, flr_d;
    std::cin >> flr_a >> flr_b >> flr_c >> flr_d;

    std::string tex_path;
    std::cin >> tex_path;
    
    float3 flr_clr;
    float flr_rfl;
    std::cin >> flr_clr >> flr_rfl;

    std::ifstream fin(tex_path, std::ios::binary);
    if (!fin.is_open()) {
        std::cout << "Cannot open texture!\n";
        exit(0);
    }

    int w_tex, h_tex;
    fin.read((char *) &w_tex, sizeof(int));
    fin.read((char *) &h_tex, sizeof(int));

    uchar4 *tex_device;
    uchar4 *tex = new uchar4[w_tex * h_tex];
    fin.read((char *) tex, sizeof(uchar4) * w_tex * h_tex);
    fin.close();

    // lights info
    size_t lights_size;
    std::cin >> lights_size;
    float3 *lights = new float3[lights_size];
    for (size_t i = 0; i < lights_size; ++i) {
        std::cin >> lights[i];

        //TODO
        float3 light_color;
        std::cin >> light_color;
    }

    // tech params
    size_t max_recursion_depth;
    std::cin >> max_recursion_depth;

    size_t ssaa_ratio;
    std::cin >> ssaa_ratio;

    // world initialization
    HittableObject **wrld;
    size_t wrld_size = N_OBJECTS + 1;
    float3 *wrld_lights;

    if (is_cpu)  {
        wrld = new HittableObject*[wrld_size];
        wrld[0] = new Cube(prm[0].rds, prm[0].cnt, prm[0].clr, prm[0].rfl, prm[0].rfr);
        wrld[1] = new Octahedron(prm[1].rds, prm[1].cnt, prm[1].clr, prm[1].rfl, prm[1].rfr);
        wrld[2] = new Dodecahedron(prm[2].rds, prm[2].cnt, prm[2].clr, prm[2].rfl, prm[2].rfr);

        //floor
        wrld[3] = new Rectangle(
            flr_a, flr_b, flr_c, flr_d,
            flr_clr, flr_rfl, 0,
            tex, w_tex, h_tex
        );

        generate_edge_lights((PlatonicSolid *) wrld[0], prm[0].nlights, &wrld, &wrld_size);
        generate_edge_lights((PlatonicSolid *) wrld[1], prm[1].nlights, &wrld, &wrld_size);
        generate_edge_lights((PlatonicSolid *) wrld[2], prm[2].nlights, &wrld, &wrld_size);

        wrld_lights = lights;
    } else {
        HittableObject ***wrld_device_ptr;
        CSC(cudaMalloc(&wrld_device_ptr, sizeof(HittableObject***)));

        size_t *wrld_device_size_ptr;
        CSC(cudaMalloc(&wrld_device_size_ptr, sizeof(size_t)));

        // device texture
        CSC(cudaMalloc(&tex_device, sizeof(uchar4) * w_tex * h_tex));
        CSC(cudaMemcpy(tex_device, tex, sizeof(uchar4) * w_tex * h_tex, cudaMemcpyHostToDevice));

        PlatonicSolidParams *p_dev;
        CSC(cudaMalloc(&p_dev, sizeof(PlatonicSolidParams) * N_OBJECTS));
        CSC(cudaMemcpy(p_dev, &prm, sizeof(PlatonicSolidParams) * N_OBJECTS, cudaMemcpyHostToDevice));

        RectangleParams rect = {
            flr_a, flr_b, flr_c, flr_d,
            flr_clr, flr_rfl,
            tex_device, w_tex, h_tex
        };
        RectangleParams *rect_device;
        CSC(cudaMalloc(&rect_device, sizeof(RectangleParams)));
        CSC(cudaMemcpy(rect_device, &rect, sizeof(RectangleParams), cudaMemcpyHostToDevice));

        init_world<<<1, 1>>>(wrld_device_ptr, wrld_device_size_ptr, p_dev, rect_device);
        CSC(cudaGetLastError());

        CSC(cudaMemcpy(&wrld, wrld_device_ptr, sizeof(HittableObject**), cudaMemcpyDeviceToHost));
        CSC(cudaMemcpy(&wrld_size, wrld_device_size_ptr, sizeof(size_t), cudaMemcpyDeviceToHost));

        CSC(cudaMalloc(&wrld_lights, sizeof(float3) * lights_size));
        CSC(cudaMemcpy(wrld_lights, lights, sizeof(float3) * lights_size, cudaMemcpyHostToDevice));

        CSC(cudaFree(wrld_device_ptr));
        CSC(cudaFree(wrld_device_size_ptr));
        CSC(cudaFree(p_dev));
        CSC(cudaFree(rect_device));
    }

    // rendering
    if (is_cpu) {
        RayTracer::cpu::Renderer renderer(W, H, fov, max_recursion_depth, ssaa_ratio);

        render_scene(
            renderer,
            wrld, wrld_size,
            wrld_lights, lights_size,
            out_path
        );
    } else {
        RayTracer::gpu::Renderer renderer(W, H, fov, max_recursion_depth, ssaa_ratio);

        render_scene(
            renderer,
            wrld, wrld_size,
            wrld_lights, lights_size,
            out_path
        );
    }

    // free resourses
    if (is_cpu) {
        for (int i = 0; i < wrld_size; ++i) {
            delete wrld[i];
        }
        delete[] wrld;
    } else {
        CSC(cudaFree(wrld_lights));
        free_world<<<1, 1>>>(wrld, wrld_size);
        CSC(cudaFree(tex_device));
    }

    delete lights;
    delete tex;

    return 0;
}
