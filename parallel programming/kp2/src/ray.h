#ifndef RAY_H
#define RAY_H

#include "vec_types.h"


class Ray
{
public:
    float3 orig;
    float3 dir;

    Ray() {};

    Ray(float3 o, float3 d)
    {
        orig = o;
        dir = d;
    }

    inline float3 param(float t) const
    {
        return orig + t * dir;
    }
};

#endif