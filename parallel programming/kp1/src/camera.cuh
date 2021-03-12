#ifndef CAMERA_H
#define CAMERA_H

#include "vec_math.cuh"
#include "ray.cuh"

#define M_PI 3.14159265358979323846

class Camera
{
public:
    const float3 up = make_float3(0, 1, 0);

    float3 eye;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;

    Camera(float3 from, float3 to, float vfov, float aspect)
    {
        float theta = vfov * M_PI / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;

        horizontal = make_float3(2 * half_width, 0, 0);
        vertical = make_float3(2 * half_height, 0, 0);

        setUp(from, to);
    }

    void setUp(float3 from, float3 to)
    {
        float half_width = len(horizontal) / 2;
        float half_height = len(vertical) / 2;

        float3 w = unit_vec(to - from);
        float3 u = unit_vec(cross(up, w));
        float3 v = cross(w, u);

        eye = from;
        lower_left_corner = eye + w - half_width * u - half_height * v;
        horizontal = 2 * half_width * u;
        vertical   = 2 * half_height * v;
    }

    __host__ __device__
    Camera& operator=(const Camera &cam)
    {
        this->eye = cam.eye;
        this->lower_left_corner = cam.lower_left_corner;
        this->horizontal = cam.horizontal;
        this->vertical = cam.vertical;

        return *this;
    }

    __host__ __device__
    Ray get_ray(float x, float y)
    {
        return Ray(eye, lower_left_corner + x * horizontal + y * vertical - eye);
    }
};

#endif