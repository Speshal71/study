#ifndef VEC_MATH_H
#define VEC_MATH_H


#include <iostream>


typedef unsigned char uchar;


struct int3
{
    int x;
    int y;
    int z;

    int3(int x, int y, int z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};


struct int2
{
    int x;
    int y;

    int2(int x, int y)
    {
        this->x = x;
        this->y = y;
    }
};


struct uchar4
{
    uchar x;
    uchar y;
    uchar z;
    uchar w;

    uchar4(uchar x = 0, uchar y = 0, uchar z = 0, uchar w = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
};


struct float2
{
    float x;
    float y;
};


struct float3
{
    float x;
    float y;
    float z;

    float3(float x = 0, float y = 0, float z = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};


std::istream &operator>>(std::istream &is, float3& vec)
{
    is >> vec.x >> vec.y >> vec.z;
    return is;
}


float3 operator+(float3 a, float3 b)
{
    return float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


float3& operator+=(float3& lhs, float3 rhs)
{
    lhs = lhs + rhs;
    return lhs;
}


float3 operator-(float3 a, float3 b)
{
    return float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


float3 operator*(float3 a, float3 b)
{
    return float3(a.x * b.x, a.y * b.y, a.z * b.z);
}


float3 operator*(float a, float3 b)
{
    return float3(a * b.x, a * b.y, a * b.z);
}


float3 operator/(float3 a, float b)
{
    return float3(a.x / b, a.y / b, a.z / b);
}


float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


float3 cross(float3 a, float3 b)
{
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}


float len(float3 a)
{
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}


float3 unit_vec(float3 a)
{
    float l = len(a);
    return float3(a.x / l, a.y / l, a.z / l);
}

#endif