#ifndef OBJECTS_H
#define OBJECTS_H

#include <algorithm>
#include <cmath>
#include <iostream>

#include "consts.h"
#include "vec_types.h"
#include "ray.h"


float clamp(float val, float low, float high)
{
    return std::max(low, std::min(val, high));
}



class LightSource
{
public:
    float3 pos;
    float3 color;

    LightSource(float3 pos, float3 color = constants::objects::DEFAULT_LIGHT_COLOR)
    {
        this->pos = pos;
        this->color = color;
    }
};


struct HitRecord 
{
    float t = constants::INF;
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

    virtual bool is_hitted(Ray& r, HitRecord& rec) = 0;

    virtual ~HittableObject() {}
};


class Sphere: public HittableObject
{
public:
    float3 center;
    float radius;

    Sphere(float3 c, float r, float3 clr, float refl, float refr, bool emit = false)
    {
        color = clr;
        reflection_ratio = refl;
        refraction_ratio = refr;
        is_emitted = emit;
        center = c;
        radius = r;
    }

    bool is_hitted(Ray& r, HitRecord& rec)
    {
        float3 oc = r.orig - center;
        float a = dot(r.dir, r.dir);
        float b = dot(oc, r.dir);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;

        if (discriminant > 0) {
            float t1 = clamp(
                (-b - sqrtf(discriminant)) / a,
                constants::objects::T_MIN, 
                constants::objects::T_MAX
            );
            float t2 = clamp(
                (-b + sqrtf(discriminant)) / a,
                constants::objects::T_MIN, 
                constants::objects::T_MAX
            );
            float t = std::min(t1, t2);

            if (t < constants::objects::T_MAX && t > constants::objects::T_MIN) {
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
    static const float EPSILON;
    float3 a, b, c;

    Triangle() {}

    Triangle(float3 a, float3 b, float3 c)
    {
        this->a = a;
        this->b = b;
        this->c = c;
    }

    bool intersected(Ray &r, HitRecord&rec, float2 &uv)
    {
        float3 ab_vec = b - a; 
        float3 ac_vec = c - a; 
        float3 pvec = cross(r.dir, ac_vec); 
        float det = dot(ab_vec, pvec);

        // ray and triangle are parallel if det is close to 0
        if (std::fabs(det) < EPSILON) {
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

        if (rec.t < constants::objects::T_MIN || rec.t > constants::objects::T_MAX) {
            return false;
        }

        // i want triangle to be illuminated from both sides
        // i should probably make this behavior optional. does it depend on if object is transparent?
        rec.normal = unit_vec(cross(ab_vec, ac_vec));
        rec.normal = (dot(r.dir, rec.normal) > 0 ? -1.f : 1.f) * rec.normal;

        rec.p = r.param(rec.t);

        return true;
    }

    float3 barycentric_to_cartesian(float2 uv)
    {
        return (1 - uv.x - uv.y) * a + uv.x * b + uv.y * c;
    }
};

const float Triangle::EPSILON = 0.0001f;


class Rectangle: public HittableObject
{
private:
    Triangle t1, t2;
    Triangle t1_tex, t2_tex;
    uchar4 *tex = nullptr;
    size_t w, h;

    inline float3 tex_color(float3 p_tex)
    {
        p_tex.x = clamp(p_tex.x, 0.f, float(w - 1));
        p_tex.y = clamp(p_tex.y, 0.f, float(h - 1));
        uchar4 tex_clr = tex[int(p_tex.y) * w + int(p_tex.x)];
        return float3(tex_clr.x, tex_clr.y, tex_clr.z) / 255;
    }

public:
    Rectangle(float3 a, float3 b, float3 c, float3 d, 
              float3 clr, float refl, float refr)
    {
        color = clr;
        reflection_ratio = refl;
        refraction_ratio = refr;

        t1 = Triangle(a, b, c);
        t2 = Triangle(c, d, a);
    }

    void set_texture(uchar4 *tex, int w, int h)
    {
        this->tex = tex;
        this->w = w;
        this->h = h;

        t1_tex = Triangle(
            float3(0, float(h - 1), 0), 
            float3(float(w - 1), float(h - 1), 0), 
            float3(float(w - 1), 0, 0)
        );
        t2_tex = Triangle(
            float3(float(w - 1), 0, 0), 
            float3(0, 0, 0), 
            float3(0, float(h - 1), 0)
        );
    }

    bool is_hitted(Ray& r, HitRecord& rec)
    {
        float2 uv;

        if (t1.intersected(r, rec, uv)) {
            rec.color = (tex ? tex_color(t1_tex.barycentric_to_cartesian(uv)) : color); //* color;
            rec.reflection_ratio = reflection_ratio;
            rec.refraction_ratio = refraction_ratio;
            return true;
        }

        if (t2.intersected(r, rec, uv)) {
            rec.color = (tex ? tex_color(t2_tex.barycentric_to_cartesian(uv)) : color); //* color;
            rec.reflection_ratio = reflection_ratio;
            rec.refraction_ratio = refraction_ratio;
            return true;
        }
        
        return false;
    }
};


class PlatonicSolid: public HittableObject
{
public:
    Triangle *t = nullptr;
    float radius;
    float3 center;

    virtual inline float base_radius() {return 0;}

    virtual inline size_t n_vertices() {return 0;}
    virtual inline float3* vertices() {return nullptr;}

    virtual inline size_t n_triangles() {return 0;}
    virtual inline int3* triangles() {return nullptr;}

    virtual inline size_t n_edges() {return 0;}
    virtual inline int2* edges() {return nullptr;}
    
    void init_triangle_mesh()
    {
        t = new Triangle[n_triangles()];

        float3 *v = vertices();
        for (int i = 0; i < n_vertices(); ++i) {
            v[i] = radius / base_radius() * v[i] + center;
        }

        int3 *t_info = triangles();
        for (int i = 0; i < n_triangles(); ++i) {
            t[i] = Triangle(
                v[t_info[i].x],
                v[t_info[i].y],
                v[t_info[i].z]
            );
        }

        delete[] v;
        delete[] t_info;
    }

    bool is_hitted(Ray& r, HitRecord& rec)
    {
        float2 uv;
        float t_closest = constants::INF;
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

    ~PlatonicSolid()
    {
        delete[] t;
    }
};


class Cube: public PlatonicSolid
{
public:
    Cube(float3 center, float radius, float3 color, float refl, float refr)
    {
        this->radius = radius;
        this->center = center;
        this->color = color;
        this->reflection_ratio = refl;
        this->refraction_ratio = refr;
        this->is_emitted = false;

        init_triangle_mesh();
    }

    inline float base_radius() {return float(std::sqrt(2));}
    inline size_t n_vertices() {return 8;}
    inline size_t n_triangles() {return 12;}
    inline size_t n_edges() {return 12;}

    inline float3* vertices()
    {
        return new float3[8]{
            {1, -1, 1}, {-1, -1, 1}, {-1, 1, 1}, {1, 1, 1}, 
            {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}
        };
    }

    inline int3* triangles()
    {
        return new int3[12]{
            {0, 1, 2}, {2, 3, 0}, {3, 4, 5}, {5, 0, 3},
            {6, 5, 4}, {4, 7, 6}, {1, 6, 7}, {7, 2, 1},
            {4, 3, 2}, {2, 7, 4}, {5, 6, 1}, {1, 0, 5}
        };
    }

    inline int2* edges()
    {
        return new int2[12]{
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, {3, 4}, {4, 5}, 
            {5, 0}, {5, 6}, {6, 7}, {7, 4}, {7, 2}, {1, 6}
        };
    }
};


class Octahedron: public PlatonicSolid
{
public:
    Octahedron(float3 center, float radius, float3 color, float refl, float refr)
    {
        this->radius = radius;
        this->center = center;
        this->color = color;
        this->reflection_ratio = refl;
        this->refraction_ratio = refr;
        this->is_emitted = false;

        init_triangle_mesh();
    }

    inline float base_radius() {return 1;}
    inline size_t n_vertices() {return 6;}
    inline size_t n_triangles() {return 8;}
    inline size_t n_edges() {return 12;}

    inline float3* vertices()
    {
        return new float3[6]{
            {0, 0, 1}, {-1, 0, 0}, {0, 0, -1},
            {1, 0, 0}, {0, 1, 0}, {0, -1, 0}
        };
    }

    inline int3* triangles()
    {
        return new int3[8]{
            {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
            {0, 5, 1}, {1, 5, 2}, {2, 5, 3}, {3, 5, 0}
        };
    }

    inline int2* edges()
    {
        return new int2[12]{
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4},
            {2, 4}, {3, 4}, {0, 5}, {1, 5}, {2, 5}, {3, 5}
        };
    }
};


class Dodecahedron: public PlatonicSolid
{
public:
    const float PHI = (1 + sqrtf(5)) / 2;

    Dodecahedron(float3 center, float radius, float3 color, float refl, float refr)
    {
        this->radius = radius;
        this->center = center;
        this->color = color;
        this->reflection_ratio = refl;
        this->refraction_ratio = refr;
        this->is_emitted = false;

        init_triangle_mesh();
    }

    inline float base_radius() {return float(std::sqrt(3));}
    inline size_t n_vertices() {return 20;}
    inline size_t n_triangles() {return 36;}
    inline size_t n_edges() {return 30;}

    inline float3* vertices()
    {
        return new float3[20]{
            {1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1}, // 0 - 3
            {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}, // 4 - 7
            {1 / PHI, PHI, 0}, {1 / PHI, -PHI, 0}, {-1 / PHI, PHI, 0}, {-1 / PHI, -PHI, 0}, // 8 - 11
            {0, 1 / PHI, PHI}, {0, 1 / PHI, -PHI}, {0, -1 / PHI, PHI}, {0, -1 / PHI, -PHI}, // 12 - 15
            {PHI, 0, 1 / PHI}, {PHI, 0, -1 / PHI}, {-PHI, 0, 1 / PHI}, {-PHI, 0, -1 / PHI} // 16 - 19
        };
    }

    inline int3* triangles()
    {
        return new int3[36]{
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
    }

    inline int2* edges()
    {
        return new int2[30]{
            {6, 11}, {11, 9}, {9, 2}, {2, 14}, {14, 6}, 
            {2, 16}, {16, 0}, {0, 12}, {12, 14}, {19, 7},
            {19, 5}, {5, 13}, {9, 3}, {3, 17}, {17, 16},
            {10, 5}, {19, 18}, {7, 15}, {15, 3}, {11, 7},
            {15, 13}, {13, 1}, {1, 17}, {1, 8}, {8, 0},
            {8, 10}, {10, 4}, {14, 12}, {4, 18}, {18, 6}
        };
    }
};


std::vector<HittableObject*> generate_edge_lights(PlatonicSolid *fig, size_t lights_per_edge)
{
    std::vector<HittableObject*> lights;

    float3 *vertices = fig->vertices();
    for (int i = 0; i < fig->n_vertices(); ++i) {
        vertices[i] = fig->radius / fig->base_radius() * vertices[i] + fig->center;
    }
    int2 *edges = fig->edges();

    for (int i = 0; i < fig->n_edges(); ++i) {
        float3 orig = vertices[edges[i].x];
        float3 dir = vertices[edges[i].y] - orig;
        for (int j = 1; j < (lights_per_edge + 1); ++j) {
            lights.push_back(
                new Sphere(
                    orig + float(j) * dir / float(lights_per_edge + 1), 
                    0.05f,
                    float3(1, 1, 1),
                    1, 1,
                    true
                )
            );
        }
    }

    delete[] vertices;
    delete[] edges;

    return lights;
}


#endif