/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "ColmapReader.h"


namespace Saiga
{
ColmapReader::ColmapReader(const std::string& dir)
{
    Load(dir);
}


void ColmapReader::Load(const std::string& dir)
{
    std::string img_file = dir + "/images.bin";
    std::string cam_file = dir + "/cameras.bin";
    std::string poi_file = dir + "/points3D.bin";

    SAIGA_ASSERT(std::filesystem::exists(img_file));
    SAIGA_ASSERT(std::filesystem::exists(cam_file));
    SAIGA_ASSERT(std::filesystem::exists(poi_file));


    col_point_to_id.clear();
    col_img_to_id.clear();
    col_cam_to_id.clear();

    col_point_to_id[colmapkInvalidPoint3DId] = colmapkInvalidPoint3DId;
    col_img_to_id[colmapkInvalidPoint2DIdx]  = colmapkInvalidPoint2DIdx;
    col_cam_to_id[colmapkInvalidPoint2DIdx]  = colmapkInvalidPoint2DIdx;


    {
        // Read cameras
        BinaryFile file(cam_file, std::ios::in);

        uint64_t num_cameras;
        file >> num_cameras;
        std::cout << "Num cameras " << num_cameras << std::endl;

        cameras.resize(num_cameras);

        for (int i = 0; i < num_cameras; ++i)
        {
            auto& c = cameras[i];
            file >> c.camera_id >> c.model_id;
            file >> c.w >> c.h;

            // update camera id
            col_cam_to_id[c.camera_id] = i;
            c.camera_id                = i;

            std::cout << "id: " << c.camera_id << " camera model: " << c.model_id << " ";
            switch (c.model_id)
            {
                case 1:
                {
                    // Pinhole
                    // fx, fy, cx, cy
                    std::array<double, 4> coeffs;
                    file >> coeffs;
                    c.K.fx = coeffs[0];
                    c.K.fy = coeffs[1];
                    c.K.cx = coeffs[2];
                    c.K.cy = coeffs[3];
                    break;
                }
                case 2:
                {
                    // Simple Radial
                    // f, cx, cy, k1
                    std::array<double, 4> coeffs;
                    file >> coeffs;
                    c.K.fx   = coeffs[0];
                    c.K.fy   = coeffs[0];
                    c.K.cx   = coeffs[1];
                    c.K.cy   = coeffs[2];
                    c.dis.k1 = coeffs[3];
                    break;
                }
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
                case 4:
                {
                    // OPENCV
                    // fx, fy, cx, cy,   k1, k2, p1, p2
                    std::array<double, 8> coeffs;
                    file >> coeffs;
                    c.K.fx = coeffs[0];
                    c.K.fy = coeffs[1];
                    c.K.cx = coeffs[2];
                    c.K.cy = coeffs[3];

                    c.dis.k1 = coeffs[4];
                    c.dis.k2 = coeffs[5];
                    c.dis.p1 = coeffs[6];
                    c.dis.p2 = coeffs[7];
                    break;
                }

                case 6:
                {
                    // FULL_OPENCV
                    // fx, fy, cx, cy,   k1, k2, p1, p2,   k3, k4, k5, k6
                    std::array<double, 12> coeffs;
                    file >> coeffs;
                    c.K.fx = coeffs[0];
                    c.K.fy = coeffs[1];
                    c.K.cx = coeffs[2];
                    c.K.cy = coeffs[3];

                    c.dis.k1 = coeffs[4];
                    c.dis.k2 = coeffs[5];
                    c.dis.p1 = coeffs[6];
                    c.dis.p2 = coeffs[7];

                    c.dis.k3 = coeffs[8];
                    c.dis.k4 = coeffs[9];
                    c.dis.k5 = coeffs[10];
                    c.dis.k6 = coeffs[11];
                    break;
                };
                default:
                    SAIGA_EXIT_ERROR(
                        "unknown camera model. checkout colmap/src/base/camera_models.h and update this file.");
            }
            std::cout << " K: " << c.K << " Dis: " << c.dis << std::endl;
        }
    }

    {
        // Read images
        BinaryFile file(img_file, std::ios::in);

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

            ci.camera_id = col_cam_to_id[ci.camera_id];

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

            images.push_back(ci);
        }


        std::sort(images.begin(), images.end(), [](auto& i1, auto& i2) { return i1.name < i2.name; });

        for (int i = 0; i < num_images; ++i)
        {
            auto& img = images[i];

            int id_before = img.image_id;

            SAIGA_ASSERT(id_before < 100000);
            col_img_to_id[img.image_id] = i;

            img.image_id = col_img_to_id[img.image_id];


            std::cout << img.name << " " << id_before << " -> " << img.image_id << " " << img.camera_id << " ";
            std::cout << " Position: " << img.t.transpose() << std::endl;
        }
    }

    {
        // Read points
        BinaryFile file(poi_file, std::ios::in);

        uint64_t num_points;
        file >> num_points;
        std::cout << "Num Point3D " << num_points << std::endl;

        points.resize(num_points);

        for (int i = 0; i < num_points; ++i)
        {
            auto& p = points[i];

            file >> p.world_point_index;

            SAIGA_ASSERT(p.world_point_index < 100000000UL);
            SAIGA_ASSERT(p.world_point_index != colmapkInvalidPoint3DId);

            col_point_to_id[p.world_point_index] = i;
            p.world_point_index                  = i;

            file >> p.position.x() >> p.position.y() >> p.position.z();
            file >> p.color.x() >> p.color.y() >> p.color.z();
            file >> p.error;

            uint64_t num_tracks;
            file >> num_tracks;

            p.tracks.resize(num_tracks);

            for (auto& t : p.tracks)
            {
                file >> t.image_id >> t.keypoint_id;

                SAIGA_ASSERT(col_img_to_id.count(t.image_id) > 0);
                t.image_id = col_img_to_id[t.image_id];
            }

            // std::cout << "Point " << p.world_point_index << " " << p.position.transpose() << " Tracks: " <<
            // num_tracks
            //           << std::endl;
        }

        for (auto& img : images)
        {
            for (auto& o : img.obvservations)
            {
                SAIGA_ASSERT(col_point_to_id.count(o.world_point_index) > 0);
                uint64_t old_index = o.world_point_index;
                SAIGA_ASSERT(col_point_to_id.count(old_index) > 0);
                o.world_point_index = col_point_to_id[o.world_point_index];

                if (old_index == colmapkInvalidPoint3DId)
                {
                    continue;
                    std::cout << "invalid obs" << std::endl;
                    SAIGA_ASSERT(o.world_point_index == old_index);
                }

                auto& wp = points[o.world_point_index];

                bool found = false;
                for (auto t : wp.tracks)
                {
                    if (t.image_id == img.image_id)
                    {
                        found = true;
                    }
                }

                if (!found)
                {
                    std::cout << "Error img/wp " << img.image_id << " wp index: " << old_index << " -> "
                              << o.world_point_index << std::endl;
                    for (auto t : wp.tracks) std::cout << t.image_id << std::endl;
                }
                SAIGA_ASSERT(found);
            }
        }
    }

    SAIGA_ASSERT(Check());
}
void ColmapReader::Save(const std::string& dir)
{
    SAIGA_ASSERT(Check());
    std::filesystem::create_directories(dir);

    std::string img_file = dir + "/images.bin";
    std::string cam_file = dir + "/cameras.bin";
    std::string poi_file = dir + "/points3D.bin";


    {
        // Read cameras
        BinaryFile file(cam_file, std::ios::out);

        uint64_t num_cameras = cameras.size();
        file << num_cameras;

        for (auto& c : cameras)
        {
            file << c.camera_id << c.model_id;
            file << c.w << c.h;


            // FULL_OPENCV
            // fx, fy, cx, cy,   k1, k2, p1, p2,   k3, k4, k5, k6
            std::array<double, 12> coeffs;
            coeffs[0] = c.K.fx;
            coeffs[1] = c.K.fy;
            coeffs[2] = c.K.cx;
            coeffs[3] = c.K.cy;

            coeffs[4] = c.dis.k1;
            coeffs[5] = c.dis.k2;
            coeffs[6] = c.dis.p1;
            coeffs[7] = c.dis.p2;

            coeffs[8]  = c.dis.k3;
            coeffs[9]  = c.dis.k4;
            coeffs[10] = c.dis.k5;
            coeffs[11] = c.dis.k6;
            file << coeffs;
        }
    }


    {
        // save images
        BinaryFile file(img_file, std::ios::out);

        uint64_t num_images = images.size();
        file << num_images;

        for (auto& ci : images)
        {
            file << ci.image_id;

            file << ci.q.w() << ci.q.x() << ci.q.y() << ci.q.z();
            file << ci.t.x() << ci.t.y() << ci.t.z();
            file << ci.camera_id;


            for (char x : ci.name)
            {
                file << x;
            }
            file << '\0';

            uint64_t num_points = ci.obvservations.size();
            file << num_points;

            for (auto& p : ci.obvservations)
            {
                file << p.keypoint.x() << p.keypoint.y();
                file << p.world_point_index;
            }
        }
    }



    {
        // save points
        BinaryFile file(poi_file, std::ios::out);

        uint64_t num_points = points.size();
        file << num_points;

        for (auto p : points)
        {
            file << p.world_point_index;
            file << p.position.x() << p.position.y() << p.position.z();
            file << p.color.x() << p.color.y() << p.color.z();
            file << p.error;

            uint64_t num_tracks = p.tracks.size();
            file << num_tracks;

            for (auto& t : p.tracks)
            {
                file << t.image_id << t.keypoint_id;
            }
        }
    }
}
bool ColmapReader::Check()
{
    for (auto& img : images)
    {
        for (auto& o : img.obvservations)
        {
            if (o.world_point_index == colmapkInvalidPoint3DId) continue;

            SAIGA_ASSERT(o.world_point_index < points.size());
            auto& wp = points[o.world_point_index];
            SAIGA_ASSERT(wp.world_point_index == o.world_point_index);



            bool found = false;
            for (auto t : wp.tracks)
            {
                // std::cout << "track " << t.image_id << " " << t.keypoint_id << std::endl;
                if (t.image_id == img.image_id)
                {
                    found = true;
                }
            }

            if (!found)
            {
                std::cout << "Error img/wp " << img.image_id << " " << wp.world_point_index << std::endl;
                //                for (auto t : wp.tracks) std::cout << t.image_id << std::endl;
            }
            //            SAIGA_ASSERT(found);
        }
    }
    return true;
}
}  // namespace Saiga
