/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/geometry/all.h"
#include "saiga/core/image/all.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/DepthmapPreprocessor.h"

#include "SparseTSDF.h"

#include <set>
namespace Saiga
{
struct SAIGA_VISION_API FusionParams
{
#if 1
    // Params for body reconstruction
    float voxelSize               = 0.003;
    float truncationDistance      = 0.01;
    float truncationDistanceScale = 0.01;
    float maxIntegrationDistance  = 2;
#else
    // Params for room reonstruction
    float voxelSize               = 0.01;
    float truncationDistance      = 0.02;
    float truncationDistanceScale = 0.02;
    float maxIntegrationDistance  = 5;
#endif
    bool use_confidence = true;

    float newWeight = 1;
    float maxWeight = 250;

    std::string out_file = "outmesh_sparse.off";

    void imgui();
};

struct SAIGA_VISION_API FusionImage
{
    // Set by the user
    ImageView<const float> depthMap;

    // Confidence for every depth value
    TemplatedImage<float> confidence;
    TemplatedImage<vec3> unprojected_position;


    SE3 V;

    std::vector<ivec3> visible_blocks;
    std::vector<ivec3> truncated_blocks;
};


struct SAIGA_VISION_API FusionScene
{
    std::vector<FusionImage> images;
    std::shared_ptr<std::vector<TemplatedImage<float>>> local_depth_images;

    ImageDimensions depth_map_size;
    double bf;
    Intrinsics4 K;
    Distortion dis;

    int Size() const { return images.size(); }


    FusionParams params;
    void imgui();
    virtual void Fuse();

   protected:
    std::shared_ptr<SparseTSDF<8>> tsdf;

    std::vector<std::array<vec3, 3>> triangle_soup;
    TriangleMesh<VertexNC, uint32_t> mesh;

    std::vector<int> triangle_soup_inclusive_prefix_sum;

    TemplatedImage<vec2> unproject_undistort_map;

    void Preprocess();
    void AnalyseSparseStructure();
    void ComputeWeight();
    void Allocate();
    void Integrate();
    void ExtractMesh();
};



}  // namespace Saiga
