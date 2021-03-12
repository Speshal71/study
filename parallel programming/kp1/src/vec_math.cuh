#ifndef VEC_MATH_H
#define VEC_MATH_H

__host__ __device__
float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
float3& operator+=(float3& lhs, float3& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

__host__ __device__
float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__
float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__
float3 operator*(float a, float3 b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__
float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__
float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
float3 cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__
float len(float3 a)
{
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__
float3 unit_vec(float3 a)
{
    float l = len(a);
    return make_float3(a.x / l, a.y / l, a.z / l);
}

#endif