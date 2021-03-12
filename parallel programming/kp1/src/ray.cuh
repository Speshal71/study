#ifndef RAY_H
#define RAY_H

#include "vec_math.cuh"

class Ray
{
public:
    float3 orig;
    float3 dir;

    __host__ __device__
    Ray() {};

    __host__ __device__
    Ray(float3 o, float3 d)
    {
        orig = o;
        dir = d;
    }

    __host__ __device__
    inline float3 param(float t) const
    {
        return orig + t * dir;
    }
};


#endif