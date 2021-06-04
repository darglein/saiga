/**
 * Copyright (c) 2021 Darius Rückert
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

#include <saiga/core/model/UnifiedMesh.h>
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
    bool use_confidence              = true;
    bool test                        = false;
    bool bilinear_intperpolation     = true;
    bool increase_visibility_frustum = false;

    // The input data is perfect (no outliers, noise)
    // -> Use a large truncation distance, but min/max the distance
    bool ground_truth_fuse          = false;
    float ground_truth_trunc_factor = 1.0f;

    // the truncation distance will be always greater than (min_truncation_factor * voxelSize)
    float min_truncation_factor = 6;

    // clamp signed distance to +- this value
    float sd_clamp = 100;

    float extract_iso            = 0;
    float extract_outlier_factor = 8;
    float newWeight              = 0.1;
    float maxWeight              = 250;

    int hash_size          = 5 * 1000 * 1000;
    int block_count        = 25 * 1000;
    bool post_process_mesh = true;

    // added to projet image points.
    // for example -0.5 for opengl renders
    Vec2 ip_offset = Vec2::Zero();


    bool verbose         = true;
    bool point_based     = false;
    std::string out_file = "outmesh_sparse.off";

    void imgui();
};

class SAIGA_VISION_API FusionImage
{
   public:
    // Set by the user
    ImageView<const float> depthMap;
    SE3 V;

    // Confidence for every depth value
    TemplatedImage<float> confidence;
    TemplatedImage<vec3> unprojected_position;

    std::vector<ivec3> visible_blocks;
    //    std::vector<ivec3> truncated_blocks;
};


struct SAIGA_VISION_API FusionScene
{
    // Set by the user
    std::vector<FusionImage> images;
    std::shared_ptr<std::vector<TemplatedImage<float>>> local_depth_images;
    IntrinsicsPinholed K;
    Distortion dis;
    FusionParams params;

    FusionScene() {}
    int Size() const { return images.size(); }
    void imgui();
    virtual void Fuse();

    void FuseIncrement(const FusionImage& image, bool first);

    ImageDimensions depth_map_size;
    std::shared_ptr<SparseTSDF> tsdf;

    std::vector<std::array<vec3, 3>> triangle_soup;
    UnifiedMesh mesh;

    std::vector<int> triangle_soup_inclusive_prefix_sum;

    TemplatedImage<vec2> unproject_undistort_map;


    void Preprocess();
    void AnalyseSparseStructure();
    void ComputeWeight();
    void Visibility();
    void Integrate();
    void IntegratePointBased();
    void ExtractMesh();
};



}  // namespace Saiga
