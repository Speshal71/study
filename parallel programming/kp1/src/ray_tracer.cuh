#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <thrust/device_vector.h>
#include <thrust/device_reference.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <curand_kernel.h>

#include "csc.cuh"
#include "vec_math.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include "objects.cuh"


#define INF 10000000
#define WHITE_COLOR make_float3(1,1,1)
#define BACKGROUND_COLOR make_float3(0,0,0)

#define LIGHTS_ILLUMINATON 0
#define REFLECTION 1
#define REFRACTION 2
#define CLEAR_STACK 3

#define TRACE_EMMITED true


struct RecursionRecord 
{
    int state;
    Ray ray;
    float3 color;
    HitRecord hit_rec;
};


__constant__ size_t max_recursion_depth_gpu;
size_t max_recursion_depth_cpu;


namespace RayTracer
{
    namespace common
    {
        __host__ __device__
        float3 clamp_color(float3 color)
        {
            return make_float3(
                fminf(color.x, 1),
                fminf(color.y, 1),
                fminf(color.z, 1)
            );
        }


        __host__ __device__
        float3 reflect(float3 incident_dir, float3 normal)
        {
            return incident_dir - 2 * dot(incident_dir, normal) * normal; 
        }


        __host__ __device__
        float3 refract(float3 incident_dir, float3 normal)
        {
            return incident_dir; 
        }


        __host__ __device__
        int trace_intersect(Ray &r, 
                            HittableObject **wrld, size_t wrld_size,
                            HitRecord &hit_rec,
                            bool trace_emit = false)
        {
            float t_closest = INF;
            int i_closest = -1;
            HitRecord temp_hit_rec;
            
            for (int i = 0; i < wrld_size; ++i) {
                if ((trace_emit || !wrld[i]->is_emitted) && wrld[i]->is_hitted(r, temp_hit_rec)) {
                    if (temp_hit_rec.t < t_closest) {
                        hit_rec = temp_hit_rec;
                        t_closest = temp_hit_rec.t;
                        i_closest = i;
                    }
                } 
            }
        
            return i_closest;
        }
    };


    namespace cpu
    {
        class Renderer
        {
        private:
            size_t W, H;
            size_t ssaa_ratio;
            Camera cam;
        
        public:
            Renderer(size_t W, size_t H, size_t fov, size_t max_recursion_depth = 5, size_t ssaa_ratio = 1):
                cam(make_float3(0, 0, 0), make_float3(0, 0, -5), 90, float(W) / float(H))
            {
                this->W = W;
                this->H = H;
                this->ssaa_ratio = ssaa_ratio;
                max_recursion_depth_cpu = max_recursion_depth;
            }

            __host__
            static float3 get_color(
                Ray &r, 
                HittableObject **wrld, size_t wrld_size, 
                float3 *lights, size_t lights_size, 
                int recursion_depth
            );

            __host__
            void render(
                HittableObject **wrld, size_t wrld_size, 
                float3 *lights, size_t lights_size, 
                uchar4 *image
            );

            __host__
            inline size_t width() {
                return W;
            }

            __host__
            inline size_t height() {
                return H;
            }

            __host__
            void setUpCam(float3 from, float3 to)
            {
                cam.setUp(from, to);
            }
        };
    };


    namespace gpu
    {
        class Renderer
        {
        private:
            size_t W, H;
            size_t ssaa_ratio;
            Camera cam;
            uchar4 *image_buffer;
            dim3 grid_dim = dim3(3, 3);
            dim3 block_dim = dim3(16, 16);
            size_t recursion_buffer_size;
            RecursionRecord *recursion_buffer;

        public:
            Renderer(size_t W, size_t H, size_t fov, size_t max_recursion_depth = 5, size_t ssaa_ratio = 1):
                cam(make_float3(0, 0, 0), make_float3(0, 0, -5), fov, float(W) / float(H))
            {
                this->W = W;
                this->H = H;
                this->ssaa_ratio = ssaa_ratio;
                CSC(cudaMalloc(&image_buffer, sizeof(uchar4) * W * H * ssaa_ratio));

                recursion_buffer_size = (
                    grid_dim.x * grid_dim.y *
                    block_dim.x * block_dim.y * 
                    (max_recursion_depth + 1)
                );

                RecursionRecord Default;
                Default.state = LIGHTS_ILLUMINATON;
                Default.color = make_float3(0, 0, 0);

                CSC(cudaMalloc(&recursion_buffer, sizeof(RecursionRecord) * recursion_buffer_size));
                //CSC(cudaMemset(recursion_buffer, 0, sizeof(RecursionRecord) * recursion_buffer_size));
                thrust::uninitialized_fill_n(
                    thrust::device_pointer_cast(recursion_buffer), 
                    recursion_buffer_size, 
                    Default
                );

                CSC(cudaMemcpyToSymbol(max_recursion_depth_gpu, &max_recursion_depth, sizeof(size_t)));
            }

            __device__
            static float3 get_color(
                Ray r, 
                HittableObject** wrld, int wrld_size, 
                float3* lights, int lights_size,
                RecursionRecord *recursion_stack
            );

            __host__
            void render(
                HittableObject **wrld, size_t wrld_size, 
                float3 *lights, size_t lights_size, 
                uchar4 *image
            );

            __host__ __device__
            inline size_t width() {
                return W;
            }

            __host__ __device__
            inline size_t height() {
                return H;
            }

            __host__
            void setUpCam(float3 from, float3 to)
            {
                cam.setUp(from, to);
            }

            ~Renderer()
            {
                CSC(cudaFree(image_buffer));
                CSC(cudaFree(recursion_buffer));
            }
        };

        __global__
        void render_kernel(
            Camera *cam, 
            HittableObject **wrld, size_t wrld_size, 
            float3 *lights, size_t lights_size,
            uchar4 *image, size_t W, size_t H,
            RecursionRecord *recursion_buffer, 
            size_t recursion_buffer_size,
            size_t ssaa_ratio
        );
    };
};


__host__
float3 RayTracer::cpu::Renderer::get_color(Ray &r, 
                                           HittableObject **wrld, size_t wrld_size, 
                                           float3 *lights, size_t lights_size, 
                                           int recursion_depth)
{
    if (recursion_depth >= max_recursion_depth_cpu) {
        return BACKGROUND_COLOR;
    }

    float3 color = BACKGROUND_COLOR;
    HitRecord hit_rec;

    int i_closest = common::trace_intersect(r, wrld, wrld_size, hit_rec, TRACE_EMMITED);
    if (i_closest != -1) {
        if (wrld[i_closest]->is_emitted) {
            return hit_rec.color;
        }
        // trace if light sources illuminate intersect point
        for (int i = 0; i < lights_size; ++i) {
            float3 light_ray_dir = lights[i] - hit_rec.p;
            float light_distance = len(light_ray_dir);
            Ray light_ray(hit_rec.p, unit_vec(light_ray_dir));
            HitRecord shadow_rec;
            shadow_rec.t = INF;
            
            // TODO hande case of transparent obstacle
            int i_obstacle = common::trace_intersect(light_ray, wrld, wrld_size, shadow_rec);
            if (i_obstacle == -1 || shadow_rec.t > light_distance) {
                float ratio = dot(light_ray.dir, hit_rec.normal);
                ratio = ratio > 0 ? ratio: 0;
                color = color + 16 * ratio * WHITE_COLOR / light_distance / light_distance;
            }
        }

        if (hit_rec.reflection_ratio > 0) {
            float3 reflected_dir = common::reflect(r.dir, hit_rec.normal);
            Ray reflected_ray(hit_rec.p, reflected_dir);

            color = color + (
                hit_rec.reflection_ratio * 
                get_color(reflected_ray, wrld, wrld_size, lights, lights_size, recursion_depth + 1)
            );
        }

        if (hit_rec.refraction_ratio > 0) {
            float3 refracted_dir = common::refract(r.dir, hit_rec.normal);
            Ray refracted_ray(hit_rec.p, refracted_dir);

            color = color + (
                hit_rec.refraction_ratio * 
                get_color(refracted_ray, wrld, wrld_size, lights, lights_size, recursion_depth + 1)
            );
        }

        color = color * hit_rec.color;
    }

    return color;
}


__host__
void RayTracer::cpu::Renderer::render(HittableObject **wrld, size_t wrld_size, 
                                      float3 *lights, size_t lights_size,  
                                      uchar4 *image)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            Ray r = cam.get_ray(float(j) / W, float(i) / H);
            float3 pixel = common::clamp_color(get_color(r, wrld, wrld_size, lights, lights_size, 0));
            image[(H - 1 - i) * W + j] = make_uchar4(
                pixel.x * 255,
                pixel.y * 255,
                pixel.z * 255,
                255
            );
        }
    }
}


__device__
float3 RayTracer::gpu::Renderer::get_color(Ray r, 
                                           HittableObject** wrld, int wrld_size, 
                                           float3* lights, int lights_size,
                                           RecursionRecord *recursion_stack)
{
    float3 returned_color = BACKGROUND_COLOR;
    int recursion_depth = 0;

    RecursionRecord *rr = &recursion_stack[0];
    rr->ray = r;

    while (recursion_depth >= 0) {
        if (recursion_depth >= max_recursion_depth_gpu) {
            returned_color = BACKGROUND_COLOR;
            --recursion_depth;
            continue;
        }

        rr = &recursion_stack[recursion_depth];

        if (rr->state == LIGHTS_ILLUMINATON) 
        {
            int i_closest = common::trace_intersect(rr->ray, wrld, wrld_size, rr->hit_rec, TRACE_EMMITED);
            if (i_closest != -1) {
                if (wrld[i_closest]->is_emitted) {
                    rr->color = rr->hit_rec.color;
                    rr->state = CLEAR_STACK;
                    continue;
                }
                // trace if light sources illuminate intersect point
                for (int i = 0; i < lights_size; ++i) {
                    float3 light_ray_dir = lights[i] - rr->hit_rec.p;
                    float light_distance = len(light_ray_dir);
                    Ray light_ray(rr->hit_rec.p, unit_vec(light_ray_dir));
                    HitRecord shadow_rec;
                    shadow_rec.t = INF;

                    // TODO hande case of transparent obstacle
                    int i_obstacle = common::trace_intersect(light_ray, wrld, wrld_size, shadow_rec);
                    if (i_obstacle == -1 || shadow_rec.t > light_distance) {
                        float ratio = dot(light_ray.dir, rr->hit_rec.normal);
                        ratio = ratio > 0 ? ratio: 0;
                        rr->color = rr->color + 16 * ratio * WHITE_COLOR / light_distance / light_distance;
                    }
                }
                rr->state = REFLECTION;
                if (rr->hit_rec.reflection_ratio > 0) {
                    float3 reflected_dir = common::reflect(rr->ray.dir, rr->hit_rec.normal);
                    recursion_stack[recursion_depth + 1].ray = Ray(rr->hit_rec.p, reflected_dir);
                    ++recursion_depth;
                }
            } else {
                returned_color = BACKGROUND_COLOR;
                --recursion_depth;
            }
            continue;
        } 
        else if (rr->state == REFLECTION) 
        {
            rr->color = rr->color + rr->hit_rec.reflection_ratio * returned_color;
            rr->state = REFRACTION;
            if (rr->hit_rec.refraction_ratio > 0) {
                float3 refracted_dir = common::refract(rr->ray.dir, rr->hit_rec.normal);
                recursion_stack[recursion_depth + 1].ray = Ray(rr->hit_rec.p, refracted_dir);
                ++recursion_depth;
            }
            continue;
        } 
        else if (rr->state == REFRACTION) 
        {
            rr->color = rr->color + rr->hit_rec.refraction_ratio * returned_color;
            rr->state = CLEAR_STACK;
            continue;
        } 
        else if (rr->state == CLEAR_STACK)
        {
            // clear current stack level
            returned_color = rr->color * rr->hit_rec.color;

            rr->state = LIGHTS_ILLUMINATON;
            rr->color = make_float3(0, 0, 0);
            --recursion_depth;
            continue;
        }
    }

    return returned_color;
}


__global__
void RayTracer::gpu::render_kernel(Camera *cam, 
                                   HittableObject **wrld, size_t wrld_size, 
                                   float3 *lights, size_t lights_size,
                                   uchar4 *image, size_t W, size_t H,
                                   RecursionRecord *recursion_buffer, 
                                   size_t recursion_buffer_size,
                                   size_t ssaa_ratio)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;
    size_t thread_id = idy * offset_x + idx; 

    curandState state;
    curand_init(thread_id, 0, 0, &state);
    
    for (int i = idx; i < H; i += offset_x) {
        for (int j = idy; j < W; j += offset_y) {
            float3 pixel = {0, 0, 0};
            for (int k = 0; k < ssaa_ratio; ++k) {
                Ray r = cam->get_ray(
                    (j + curand_uniform(&state)) / W, 
                    (i + curand_uniform(&state)) / H
                );
                pixel = pixel + common::clamp_color(
                    Renderer::get_color(
                        r, 
                        wrld, wrld_size, 
                        lights, lights_size, 
                        &recursion_buffer[thread_id * (max_recursion_depth_gpu + 1)]
                    )
                );
            }
            pixel = pixel / ssaa_ratio;
            image[(H - 1 - i) * W + j] = make_uchar4(
                pixel.x * 255,
                pixel.y * 255,
                pixel.z * 255,
                255
            );
        }
    }
}


__host__
void RayTracer::gpu::Renderer::render(HittableObject** wrld, size_t wrld_size, 
                                      float3* lights, size_t lights_size,  
                                      uchar4* image)
{
    thrust::device_vector<Camera> cam_device(1, cam);

    render_kernel<<<grid_dim, block_dim>>>(
        thrust::raw_pointer_cast(cam_device.data()), 
        wrld, wrld_size, 
        lights, lights_size, 
        image_buffer, W, H, 
        recursion_buffer, recursion_buffer_size,
        ssaa_ratio
    );
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(image, image_buffer, sizeof(uchar4) * W * H, cudaMemcpyDeviceToHost));
}


#endif