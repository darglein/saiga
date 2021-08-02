/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/BinaryFile.h"
#include "saiga/vision/VisionTypes.h"
namespace Saiga
{
struct ColmapReader
{
    // the colmap sfm directory must be given
    // Expects the following files inside this directory
    //    cameras.bin
    //    images.bin
    //    points3D.bin
    //
    // This is based on the actual output code of
    // colmap/src/base/reconstruction.cc
    ColmapReader(const std::string& dir)
    {
        {
            // Read images
            BinaryFile file(dir + "/images.bin", std::ios::in);

            uint64_t num_images;
            file >> num_images;
            std::cout << "Num images " << num_images << std::endl;

            for (int i = 0; i < num_images; ++i)
            {
                ColmapImage ci;

                file >> ci.image_id;

                file >> ci.q.w() >> ci.q.x() >> ci.q.y() >> ci.q.z();
                file >> ci.t;
                file >> ci.camera_id;

                char name_char;
                do
                {
                    file >> name_char;
                    if (name_char != '\0')
                    {
                        ci.name += name_char;
                    }
                } while (name_char != '\0');


                uint64_t num_points;
                file >> num_points;

                ci.obvservations.resize(num_points);

                for (auto& p : ci.obvservations)
                {
                    file >> p.keypoint.x() >> p.keypoint.y();
                    file >> p.world_point_index;
                }

                std::cout << ci.name << " " << ci.image_id << " " << ci.camera_id << std::endl;
                std::cout << ci.q << " | " << ci.t.transpose() << std::endl;

                images.push_back(ci);
            }
        }

        {
            // Read images
            BinaryFile file(dir + "/cameras.bin", std::ios::in);

            uint64_t num_cameras;
            file >> num_cameras;
            std::cout << "Num cameras " << num_cameras << std::endl;

            cameras.resize(num_cameras);

            for (auto& c : cameras)
            {
                file >> c.camera_id >> c.model_id;
                file >> c.w >> c.h;

                std::cout << "camera model: " << c.model_id << std::endl;
                switch (c.model_id)
                {
                    case 3:
                    {
                        // RADIAL
                        // f, cx, cy, k1, k2
                        std::array<double, 5> coeffs;
                        file >> coeffs;
                        c.K.fx   = coeffs[0];
                        c.K.fy   = coeffs[0];
                        c.K.cx   = coeffs[1];
                        c.K.cy   = coeffs[2];
                        c.dis.k1 = coeffs[3];
                        c.dis.k2 = coeffs[4];
                        break;
                    }
                    default:
                        SAIGA_EXIT_ERROR("unknown camera model. checkout colmap/src/base/camera_models.h");
                }
            }
        }
    }

    struct ColmapCamera
    {
        IntrinsicsPinholed K;
        Distortion dis;
        int camera_id;
        int model_id;
        uint64_t w, h;
    };

    struct ColmapImage
    {
        std::string name;
        uint32_t image_id;
        uint32_t camera_id;
        Vec3 t;
        Quat q;
        struct Obs
        {
            Vec2 keypoint;
            uint64_t world_point_index;
        };
        std::vector<Obs> obvservations;
    };
    std::vector<ColmapImage> images;
    std::vector<ColmapCamera> cameras;
};
}  // namespace Saiga
