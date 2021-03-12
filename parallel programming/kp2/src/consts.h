#ifndef CONSTS_H
#define CONSTS_H

#include "vec_types.h"


namespace constants
{
    const double PI = 3.14159265358979323846;
    const float INF = 1e+7;

    namespace ray_tracer
    {
        const float3 BACKGROUND_COLOR = float3(0, 0, 0);
        const int MAX_RECURSION_DEPTH = 5;
    }

    namespace objects
    {
        const float T_MAX = INF;
        const float T_MIN = 0.001f;
        const float3 DEFAULT_LIGHT_COLOR = float3(1, 1, 1);
        enum {
            SPHERE = 1, 
            CUBE = 2, 
            OCTAHEDRON = 3, 
            DODECAHEDRON = 4
        };
    }
}


#endif

