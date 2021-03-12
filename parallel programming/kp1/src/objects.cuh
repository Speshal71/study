#ifndef OBJECTS_H
#define OBJECTS_H

#include "vec_math.cuh"
#include "ray.cuh"
#include "csc.cuh"


#define INF 10000000
#define T_MIN 0.001f
#define T_MAX 10000000
#define epsilon 1e-4


struct HitRecord 
{
    float t;
    float3 p;
    float3 normal;
    float3 color;
    float reflection_ratio;
    float refraction_ratio;
};


class HittableObject 
{
public:
    float reflection_ratio;
    float refraction_ratio;
    float3 color;
    bool is_emitted = false;

    __host__ __device__
    virtual bool is_hitted(Ray& r, HitRecord& rec) = 0;

    __host__ __device__
    virtual ~HittableObject() {} 
};


class Sphere: public HittableObject
{
public:
    float3 center;
    float radius;

    __host__ __device__
    Sphere(float3 c, float r, float3 clr, float refl, float refr, bool emit = false)
    {
        color = clr;
        reflection_ratio = refl;
        refraction_ratio = refr;
        is_emitted = emit;
        center = c;
        radius = r;
    }

    __host__ __device__
    bool is_hitted(Ray& r, HitRecord& rec)
    {
        float3 oc = r.orig - center;
        float a = dot(r.dir, r.dir);
        float b = dot(oc, r.dir);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;

        if (discriminant > 0) {
            float t1 = fmaxf(T_MIN , fminf(T_MAX, (-b - sqrtf(discriminant)) / a));
            float t2 = fmaxf(T_MIN , fminf(T_MAX, (-b + sqrtf(discriminant)) / a));
            float t = fminf(t1, t2);

            if (t < T_MAX && t > T_MIN) {
                rec.t = t;
                rec.p = r.param(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.color = color;
                rec.reflection_ratio = reflection_ratio;
                rec.refraction_ratio = refraction_ratio;
                return true;
            }
        }
        return false;
    }
};


class Triangle
{
public:
    float3 a, b, c;

    __host__ __device__
    Triangle() {}

    __host__ __device__
    Triangle(float3 a, float3 b, float3 c)
    {
        this->a = a;
        this->b = b;
        this->c = c;
    }

    __host__ __device__
    bool intersected(Ray &r, HitRecord&rec, float2 &uv)
    {
        float3 ab_vec = b - a; 
        float3 ac_vec = c - a; 
        float3 pvec = cross(r.dir, ac_vec); 
        float det = dot(ab_vec, pvec);

        // ray and triangle are parallel if det is close to 0
        if (fabs(det) < epsilon) {
            return false;
        }

        float3 tvec = r.orig - a; 
        uv.x = dot(tvec, pvec) / det;
        if (uv.x < 0 || uv.x > 1) {
            return false;
        } 

        float3 qvec = cross(tvec, ab_vec);  
        uv.y = dot(r.dir, qvec) / det; 
        if (uv.y < 0 || (uv.x + uv.y) > 1) {
            return false;
        } 

        rec.t = dot(ac_vec, qvec) / det; 

        if (rec.t < T_MIN || rec.t > T_MAX) {
            return false;
        }

        // i want triangle to be illuminated from both sides
        // i should probably make this behavior optional. does it depend on if object is transparent?
        rec.normal = unit_vec(cross(ab_vec, ac_vec));
        rec.normal = (dot(r.dir, rec.normal) > 0 ? -1 : 1) * rec.normal;

        rec.p = r.param(rec.t);

        return true;
    }

    __host__ __device__
    float3 barycentric_to_cartesian(float2 uv)
    {
        return (1 - uv.x - uv.y) * a + uv.x * b + uv.y * c;
    }
};


class Rectangle: public HittableObject
{
private:
    Triangle t1, t2;
    Triangle t1_tex, t2_tex;
    uchar4 *tex;
    size_t w, h;

    __host__ __device__
    inline float3 tex_color(float3 p_tex)
    {
        p_tex.x = fmaxf(0, fminf(w - 1, p_tex.x));
        p_tex.y = fmaxf(0, fminf(h - 1, p_tex.y));
        uchar4 tex_clr = tex[int(p_tex.y) * w + int(p_tex.x)];
        return make_float3(tex_clr.x, tex_clr.y, tex_clr.z) / 255;
    }

public:  
    __host__ __device__
    Rectangle(float3 a, float3 b, float3 c, float3 d, 
              float3 clr, float refl, float refr, 
              uchar4 *tex = nullptr, int w = 0, int h = 0)
    {
        color = clr;
        reflection_ratio = refl;
        refraction_ratio = refr;

        t1 = Triangle(a, b, c);
        t2 = Triangle(c, d, a);

        this->tex = tex;
        this->w = w;
        this->h = h;
        t1_tex = Triangle(
            make_float3(0, h - 1, 0), 
            make_float3(w - 1, h - 1, 0), 
            make_float3(w - 1, 0, 0)
        );
        t2_tex = Triangle(
            make_float3(w - 1, 0, 0), 
            make_float3(0, 0, 0), 
            make_float3(0, h - 1, 0)
        );
    }

    __host__ __device__
    bool is_hitted(Ray& r, HitRecord& rec)
    {
        float2 uv;

        if (t1.intersected(r, rec, uv)) {
            rec.color = (tex ? tex_color(t1_tex.barycentric_to_cartesian(uv)) : color);
            rec.reflection_ratio = reflection_ratio;
            rec.refraction_ratio = refraction_ratio;
            return true;
        }

        if (t2.intersected(r, rec, uv)) {
            rec.color = (tex ? tex_color(t2_tex.barycentric_to_cartesian(uv)) : color);
            rec.reflection_ratio = reflection_ratio;
            rec.refraction_ratio = refraction_ratio;
            return true;
        }
        
        return false;
    }
};


/*
 My implementation of platonic solids is far from being good.
 Verticies and triangles should be static class constants.
 I implemented it this way because i want them to be allocated
 on device memory in order to virtual function is_hitted being able on gpu.
 So it's a cpu/gpu compatability decision.
*/


class PlatonicSolid: public HittableObject
{
public:
    Triangle *t = nullptr;
    float radius;
    float3 center;

    __host__ __device__
    virtual inline size_t n_vertices() {return 0;}

    __host__ __device__
    virtual inline float3* vertices() {return 0;}

    __host__ __device__
    virtual inline size_t n_triangles() {return 0;}

    __host__ __device__
    virtual inline int3* triangles() {return 0;}

    __host__ __device__
    virtual inline size_t n_edges() {return 0;}

    __host__ __device__
    virtual inline int2* edges() {return 0;}
    
    __host__ __device__ 
    void init_triangle_mesh()
    {
        float3 *v = vertices();
        int3 *t_info = triangles();

        t = new Triangle[n_triangles()];
        for (int i = 0; i < n_triangles(); ++i) {
            t[i] = Triangle(
                v[t_info[i].x],
                v[t_info[i].y],
                v[t_info[i].z]
            );
        }

        delete v;
        delete t_info;
    }

    __host__ __device__
    bool is_hitted(Ray& r, HitRecord& rec)
    {
        float2 uv;
        float t_closest = INF;
        int i_closest = -1;

        for (int i = 0; i < n_triangles(); ++i) {
            if (t[i].intersected(r, rec, uv) && rec.t < t_closest) {
                i_closest = i;
                t_closest = rec.t;
            }
        }

        if (i_closest != -1) {
            t[i_closest].intersected(r, rec, uv);
            rec.color = color;
            rec.reflection_ratio = reflection_ratio;
            rec.refraction_ratio = refraction_ratio;
            return true;
        }

        return false;
    }

    __host__ __device__
    ~PlatonicSolid()
    {
        delete t;
    }
};


class Cube: public PlatonicSolid
{
public:
    __host__ __device__
    Cube(float rds, float3 cntr, float3 clr, float refl, float refr)
    {
        radius = rds;
        center = cntr;
        color = clr;
        reflection_ratio = refl;
        refraction_ratio = refr;
        is_emitted = false;

        init_triangle_mesh();
    }

    __host__ __device__
    inline size_t n_vertices() {return 8;}

    __host__ __device__
    inline size_t n_triangles() {return 12;}

    __host__ __device__
    inline size_t n_edges() {return 12;}

    __host__ __device__
    inline float3* vertices()
    {
        float3 vertices[] = {
            {1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {1, 1, 1}, 
            {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}
        };

        float3 *v = new float3[8];
        for (int i = 0; i < 8; ++i) {
            v[i] = radius / sqrtf(2) * vertices[i] + center;
        }

        return v;
    }

    __host__ __device__
    inline int3* triangles()
    {
        int3 triangles[] = {
            {0, 1, 2}, {2, 3, 0}, {3, 4, 5}, {5, 0, 3},
            {6, 5, 4}, {4, 7, 6}, {1, 6, 7}, {7, 2, 1},
            {4, 3, 2}, {2, 7, 4}, {5, 6, 1}, {1, 0, 5}
        };

        int3 *t = new int3[12];
        for (int i = 0; i < 12; ++i) {
            t[i] = triangles[i];
        }

        return t;
    }

    __host__ __device__
    inline int2* edges()
    {
        int2 edges[] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, {3, 4}, {4, 5}, 
            {5, 0}, {5, 6}, {6, 7}, {7, 4}, {7, 2}, {1, 6}
        };

        int2 *e = new int2[12];
        for (int i = 0; i < 12; ++i) {
            e[i] = edges[i];
        }

        return e;
    }
};


class Octahedron: public PlatonicSolid
{
public:
    __host__ __device__
    Octahedron(float rds, float3 cntr, float3 clr, float refl, float refr)
    {
        radius = rds;
        center = cntr;
        color = clr;
        reflection_ratio = refl;
        refraction_ratio = refr;
        is_emitted = false;

        init_triangle_mesh();
    }

    __host__ __device__
    inline size_t n_vertices() {return 6;}

    __host__ __device__
    inline size_t n_triangles() {return 8;}

    __host__ __device__
    inline size_t n_edges() {return 12;}

    __host__ __device__
    inline float3* vertices()
    {
        float3 vertices[] = {
            {0, 0, 1}, {-1, 0, 0}, {0, 0, -1},
            {1, 0, 0}, {0, 1, 0}, {0, -1, 0}
        };

        float3 *v = new float3[6];
        for (int i = 0; i < 6; ++i) {
            v[i] = radius * vertices[i] + center;
        }

        return v;
    }

    __host__ __device__
    inline int3* triangles()
    {
        int3 triangles[] = {
            {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
            {0, 5, 1}, {1, 5, 2}, {2, 5, 3}, {3, 5, 0}
        };

        int3 *t = new int3[8];
        for (int i = 0; i < 8; ++i) {
            t[i] = triangles[i];
        }

        return t;
    }

    __host__ __device__
    inline int2* edges()
    {
        int2 edges[] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4},
            {2, 4}, {3, 4}, {0, 5}, {1, 5}, {2, 5}, {3, 5}
        };

        int2 *e = new int2[12];
        for (int i = 0; i < 12; ++i) {
            e[i] = edges[i];
        }

        return e;
    }
};


class Dodecahedron: public PlatonicSolid
{
public:
    const float PHI = (1 + sqrtf(5)) / 2;

    __host__ __device__
    Dodecahedron(float rds, float3 cntr, float3 clr, float refl, float refr)
    {
        radius = rds;
        center = cntr;
        color = clr;
        reflection_ratio = refl;
        refraction_ratio = refr;
        is_emitted = false;

        init_triangle_mesh();
    }

    __host__ __device__
    inline size_t n_vertices() {return 20;}

    __host__ __device__
    inline size_t n_triangles() {return 36;}

    __host__ __device__
    inline size_t n_edges() {return 30;}

    __host__ __device__
    inline float3* vertices()
    {
        float3 vertices[] = {
            {1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1}, // 0 - 3
            {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}, // 4 - 7
            {1 / PHI, PHI, 0}, {1 / PHI, -PHI, 0}, {-1 / PHI, PHI, 0}, {-1 / PHI, -PHI, 0}, // 8 - 11
            {0, 1 / PHI, PHI}, {0, 1 / PHI, -PHI}, {0, -1 / PHI, PHI}, {0, -1 / PHI, -PHI}, // 12 - 15
            {PHI, 0, 1 / PHI}, {PHI, 0, -1 / PHI}, {-PHI, 0, 1 / PHI}, {-PHI, 0, -1 / PHI} // 16 - 19
        };

        float3 *v = new float3[20];
        for (int i = 0; i < 20; ++i) {
            v[i] = radius / sqrtf(3) * vertices[i] + center;
        }

        return v;
    }

    __host__ __device__
    inline int3* triangles()
    {
        int3 triangles[] = {
            {6, 11, 9}, {6, 9, 2}, {6, 2, 14},
            {14, 2, 16}, {14, 16, 0}, {14, 0, 12},
            {9, 3, 17}, {9, 17, 16}, {9, 16, 2},
            {7, 15, 3}, {7, 3, 9}, {7, 9, 11},
            {17, 3, 15}, {17, 15, 13}, {17, 13, 1},
            {16, 17, 1}, {16, 1, 8}, {16, 8, 0},
            {12, 0, 8}, {12, 8, 10}, {12, 10, 4},
            {12, 4, 18}, {12, 18, 6}, {12, 6, 14},
            {19, 7, 11}, {19, 11, 6}, {19, 6, 18},
            {7, 19, 5}, {7, 5, 13}, {7, 13, 15},
            {4, 10, 5}, {4, 5, 19}, {4, 19, 18},
            {10, 8, 1}, {10, 1, 13}, {10, 13, 5}
        };

        int3 *t = new int3[36];
        for (int i = 0; i < 36; ++i) {
            t[i] = triangles[i];
        }

        return t;
    }

    __host__ __device__
    inline int2* edges()
    {
        int2 edges[] = {
            {6, 11}, {11, 9}, {9, 2}, {2, 14}, {14, 6}, 
            {2, 16}, {16, 0}, {0, 12}, {12, 14}, {19, 7},
            {19, 5}, {5, 13}, {9, 3}, {3, 17}, {17, 16},
            {10, 5}, {19, 18}, {7, 15}, {15, 3}, {11, 7},
            {15, 13}, {13, 1}, {1, 17}, {1, 8}, {8, 0},
            {8, 10}, {10, 4}, {14, 12}, {4, 18}, {18, 6}
        };

        int2 *e = new int2[30];
        for (int i = 0; i < 30; ++i) {
            e[i] = edges[i];
        }

        return e;
    }
};

#endif