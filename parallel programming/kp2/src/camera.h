#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>

#include "consts.h"
#include "vec_types.h"
#include "ray.h"


class Camera
{
public:
    const float3 up = float3(0, 1, 0);

    float3 eye;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;

    Camera() {}

    Camera(float3 from, float3 to, float vfov, float aspect)
    {
        float theta = float(vfov * constants::PI / 180);
        float half_height = std::tan(theta / 2);
        float half_width = aspect * half_height;

        horizontal = float3(2 * half_width, 0, 0);
        vertical = float3(2 * half_height, 0, 0);

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

    Ray get_ray(float x, float y)
    {
        return Ray(eye, lower_left_corner + x * horizontal + y * vertical - eye);
    }
};

#endif