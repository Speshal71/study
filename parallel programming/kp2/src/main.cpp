#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>

#include "camera.h"
#include "objects.h"
#include "ray_tracer.h"


int main(int argc, char *argv[])
{
    bool is_screenshot = false;
    bool is_sequantial = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--seq") {
            is_sequantial = true;
        }
        else if (arg == "--photo") {
            is_screenshot = true;
        }
    }
    
    std::string path_out;
    std::cin >> path_out;

    int nframes;
    std::cin >> nframes;

    size_t H, W;
    float vfov;
    float3 cam_pos, cam_look_at;
    std::cin >> W >> H >> vfov >> cam_pos >> cam_look_at;

    Camera cam(cam_pos, cam_look_at, vfov, float(W) / float(H));
    
    float3 a, b, c, d;
    float3 color;
    float reflection_ratio;
    float refraction_ratio;
    std::cin >> a >> b >> c >> d >> color >> reflection_ratio >> refraction_ratio;

    Rectangle *floor = new Rectangle(a, b, c, d, color, reflection_ratio, refraction_ratio);
    
    std::string texture_path;
    std::cin >> texture_path;
    std::ifstream texture_stream(texture_path, std::ofstream::binary);

    int tex_w, tex_h;
    texture_stream.read((char *) &tex_w, sizeof(int));
    texture_stream.read((char *) &tex_h, sizeof(int));

    uchar4 *texture = new uchar4[tex_w * tex_h]; 
    texture_stream.read((char *) texture, sizeof(uchar4) * tex_w * tex_h);
    texture_stream.close();
    floor->set_texture(texture, tex_w, tex_h);

    int world_size;
    std::vector<HittableObject*> world;
    world.push_back(floor);
    std::cin >> world_size;
    for (int i = 0; i < world_size; i++) {
        int type;
        float3 center;
        float radius;
        int lights_per_edge;

        std::cin >> type >> center >> radius >> color >> reflection_ratio >> refraction_ratio >> lights_per_edge;
        if (type == constants::objects::SPHERE) {
            world.push_back(new Sphere(center, radius, color, reflection_ratio, refraction_ratio));
        }
        else if (type == constants::objects::CUBE) {
            HittableObject *obj = new Cube(center, radius, color, reflection_ratio, refraction_ratio);
            world.push_back(obj);
            auto edge_lights = generate_edge_lights((PlatonicSolid *) obj, lights_per_edge);
            world.insert(world.end(), edge_lights.begin(), edge_lights.end());
        }
        else if (type == constants::objects::OCTAHEDRON) {
            HittableObject *obj = new Octahedron(center, radius, color, reflection_ratio, refraction_ratio);
            world.push_back(obj);
            auto edge_lights = generate_edge_lights((PlatonicSolid *) obj, lights_per_edge);
            world.insert(world.end(), edge_lights.begin(), edge_lights.end());
        }
        else if (type == constants::objects::DODECAHEDRON) {
            HittableObject *obj = new Dodecahedron(center, radius, color, reflection_ratio, refraction_ratio);
            world.push_back(obj);
            auto edge_lights = generate_edge_lights((PlatonicSolid *) obj, lights_per_edge);
            world.insert(world.end(), edge_lights.begin(), edge_lights.end());
        }
    }
    
    int lights_size;
    std::vector<LightSource> lights;
    std::cin >> lights_size;
    for (int i = 0; i < lights_size; i++) {
        float3 center;
        std::cin >> center >>  color;
        lights.push_back(LightSource(center, color));
    }

    uchar4 *image = new uchar4[W * H];
    RayTracer renderer;

    std::chrono::steady_clock::time_point abs_start = std::chrono::steady_clock::now();

    if (is_screenshot) {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        renderer.render(
            image, H, W,
            cam, 
            world,
            lights,
            is_sequantial
        );

        std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();

        std::cout << "frame time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()  / double(1000) << std::endl;

        std::ofstream fout("out.data", std::ofstream::binary);

        fout.write((char *) &W, sizeof(int));
        fout.write((char *) &H, sizeof(int));
    
        fout.write((char *) image, sizeof(uchar4) * W * H);
        fout.close();
    } else {
        for (int i = 0; i < nframes; ++i) {
        
            cam_pos = float3(
                float(5 * sin(2 * constants::PI * i / nframes)),
                0,
                float(-5 + 5 * cos(2 * constants::PI * i / nframes))
            );
            cam_look_at = float3(0, 0, -5);
    
            cam.setUp(cam_pos, cam_look_at);
    
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

            renderer.render(
                image, H, W,
                cam, 
                world,
                lights,
                is_sequantial
            );

            std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();

            std::cout << "frame " << i << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()  / double(1000) << std::endl;
    
            char picname[50];
            sprintf(picname, path_out.data(), i);
    
            std::ofstream fout(picname, std::ofstream::binary);
    
            fout.write((char *) &W, sizeof(int));
            fout.write((char *) &H, sizeof(int));
    
            fout.write((char *) image, sizeof(uchar4) * W * H);
            fout.close();
        }
    }

    std::chrono::steady_clock::time_point abs_finish = std::chrono::steady_clock::now();

    std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(abs_finish - abs_start).count()  / double(1000) << std::endl;

    for (int i = 0; i < world_size; ++i) {
        delete world[i];
    }

    delete[] texture;
    delete[] image;

    return 0;
}
