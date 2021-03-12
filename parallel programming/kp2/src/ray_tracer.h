#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <vector>
#include <algorithm>
#include <omp.h>

#include "vec_types.h"
#include "ray.h"
#include "camera.h"
#include "objects.h"


#define TRACE_EMMITED true


class RayTracer
{
private:
    float3 clamp_color(float3 color);
    float3 reflect(float3 incident_dir, float3 normal);
    float3 refract(float3 incident_dir, float3 normal);

    int trace_intersect(
        Ray &r, 
        HittableObject **wrld, size_t wrld_size,
        HitRecord &hit_rec,
        bool trace_emit = false
    );

    float3 get_color(
        Ray &r, 
        HittableObject **wrld, size_t wrld_size, 
        LightSource *lights, size_t lights_size, 
        int recursion_depth
    );

    void render_sequantially(
        uchar4 *image, size_t H, size_t W,
        Camera &cam, 
        HittableObject** world, size_t world_size,
        LightSource* lights, size_t lights_size
    );

    void render_parallel(
        uchar4 *image, size_t H, size_t W,
        Camera &cam, 
        HittableObject** world, size_t world_size,
        LightSource* lights, size_t lights_size
    );

public:
    void render(
        uchar4 *image, size_t H, size_t W,
        Camera &cam, 
        std::vector<HittableObject*>& world,
        std::vector<LightSource>& lights,
        bool sequentially
    );
};


float3 RayTracer::clamp_color(float3 color)
{
    return float3(
        std::min(color.x, 1.f),
        std::min(color.y, 1.f),
        std::min(color.z, 1.f)
    );
}


float3 RayTracer::reflect(float3 incident_dir, float3 normal)
{
    return incident_dir - 2 * dot(incident_dir, normal) * normal; 
}


float3 RayTracer::refract(float3 incident_dir, float3 normal)
{
    return incident_dir; 
}


int RayTracer::trace_intersect(Ray &r, 
                               HittableObject **wrld, size_t wrld_size,
                               HitRecord &hit_rec,
                               bool trace_emit)
{
    float t_closest = constants::INF;
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


float3 RayTracer::get_color(Ray &r, 
                            HittableObject **world, size_t world_size, 
                            LightSource *lights, size_t lights_size, 
                            int recursion_depth)
{
    if (recursion_depth >= constants::ray_tracer::MAX_RECURSION_DEPTH) {
        return constants::ray_tracer::BACKGROUND_COLOR;
    }

    float3 color = constants::ray_tracer::BACKGROUND_COLOR;
    HitRecord hit_rec;

    int i_closest = trace_intersect(r, world, world_size, hit_rec, TRACE_EMMITED);
    if (i_closest != -1) {
        if (world[i_closest]->is_emitted) {
            return hit_rec.color;
        }

        // trace if light sources illuminate intersect point
        for (int i = 0; i < lights_size; ++i) {
            float3 light_ray_dir = lights[i].pos - hit_rec.p;
            float light_distance = len(light_ray_dir);
            Ray light_ray(hit_rec.p, unit_vec(light_ray_dir));

            HitRecord shadow_rec;
            int i_obstacle = trace_intersect(light_ray, world, world_size, shadow_rec);
            if (i_obstacle == -1 || shadow_rec.t > light_distance) {
                float ratio = dot(light_ray.dir, hit_rec.normal);
                ratio = ratio > 0 ? ratio: 0;
                color += 16 * ratio * lights[i].color / light_distance / light_distance;
            }
        }

        if (hit_rec.reflection_ratio > 0) {
            float3 reflected_dir = reflect(r.dir, hit_rec.normal);
            Ray reflected_ray(hit_rec.p, reflected_dir);

            color += (
                hit_rec.reflection_ratio * 
                get_color(reflected_ray, world, world_size, lights, lights_size, recursion_depth + 1)
            );
        }

        if (hit_rec.refraction_ratio > 0) {
            float3 refracted_dir = refract(r.dir, hit_rec.normal);
            Ray refracted_ray(hit_rec.p, refracted_dir);

            color += (
                hit_rec.refraction_ratio * 
                get_color(refracted_ray, world, world_size, lights, lights_size, recursion_depth + 1)
            );
        }

        color = color * hit_rec.color;
    }

    return color;
}


void RayTracer::render_sequantially(uchar4 *image, size_t H, size_t W,
                                    Camera &cam, 
                                    HittableObject** world, size_t world_size,
                                    LightSource* lights, size_t lights_size)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            Ray r = cam.get_ray(float(j) / W, float(i) / H);
            float3 pixel = clamp_color(
                get_color(r, world, world_size, lights, lights_size, 0)
            );
            image[(H - 1 - i) * W + j] = uchar4(
                uchar(pixel.x * 255),
                uchar(pixel.y * 255),
                uchar(pixel.z * 255),
                uchar(255)
            );
        }
    }
}


void RayTracer::render_parallel(uchar4 *image, size_t H, size_t W,
                                Camera &cam, 
                                HittableObject** world, size_t world_size,
                                LightSource* lights, size_t lights_size)
{   
    Camera cam_copy(cam);
    #pragma omp parallel for shared(image, world, lights) firstprivate(H, W, world_size, lights_size, cam_copy) num_threads(16)
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            Ray r = cam_copy.get_ray(float(j) / W, float(i) / H);
            float3 pixel = clamp_color(
                get_color(r, world, world_size, lights, lights_size, 0)
            );
            image[(H - 1 - i) * W + j] = uchar4(
                uchar(pixel.x * 255),
                uchar(pixel.y * 255),
                uchar(pixel.z * 255),
                uchar(255)
            );
        }
    }
}


void RayTracer::render(uchar4 *image, size_t H, size_t W,
                       Camera &cam, 
                       std::vector<HittableObject*>& world,
                       std::vector<LightSource>& lights,
                       bool sequentially)
{
    HittableObject **world_ptr = world.data();
    size_t world_size = world.size();
    LightSource *lights_ptr = lights.data();
    size_t lights_size = lights.size();

    if (sequentially) {
        render_sequantially(
            image, H, W,
            cam, 
            world_ptr, world_size,
            lights_ptr, lights_size
        );
    } else {
        render_parallel(
            image, H, W,
            cam, 
            world_ptr, world_size,
            lights_ptr, lights_size
        );
    }
}


#endif