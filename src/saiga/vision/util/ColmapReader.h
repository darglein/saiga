/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/BinaryFile.h"
#include "saiga/vision/VisionTypes.h"

#include <filesystem>
#include <map>

namespace Saiga
{
const uint32_t colmapkInvalidPoint2DIdx = std::numeric_limits<uint32_t>::max();
const uint64_t colmapkInvalidPoint3DId  = std::numeric_limits<uint64_t>::max();

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

    // World -> View transformation (view matrix)
    Vec3 t;
    Quat q;
    struct Obs
    {
        Vec2 keypoint;
        uint64_t world_point_index = colmapkInvalidPoint3DId;
    };
    std::vector<Obs> obvservations;
};

struct ColmapWorldpoint
{
    uint64_t world_point_index = colmapkInvalidPoint3DId;
    Vec3 position;
    ucvec3 color;
    double error;

    struct Track
    {
        uint32_t image_id;
        uint32_t keypoint_id;
    };

    std::vector<Track> tracks;
};

struct SAIGA_VISION_API ColmapReader
{
    // the colmap sfm directory must be given
    // Expects the following files inside this directory
    //    cameras.bin
    //    images.bin
    //    points3D.bin
    //
    // This is based on the actual output code of
    // colmap/src/base/reconstruction.cc
    ColmapReader(const std::string& dir);
    ColmapReader() {}

    ColmapReader(const std::vector<ColmapCamera>& cameras, const std::vector<ColmapImage>& images,
                 const std::vector<ColmapWorldpoint>& points)
        : cameras(cameras), images(images), points(points)
    {
    }


    void Load(const std::string& dir);
    void Save(const std::string& dir);


    bool Check();

    std::vector<ColmapCamera> cameras;
    std::vector<ColmapImage> images;
    std::vector<ColmapWorldpoint> points;


    std::map<uint32_t, uint32_t> col_cam_to_id;
    std::map<uint32_t, uint32_t> col_img_to_id;
    std::map<uint64_t, uint64_t> col_point_to_id;
};
}  // namespace Saiga
