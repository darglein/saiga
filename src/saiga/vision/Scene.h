/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/image/image.h"
#include "saiga/util/statistics.h"
#include "saiga/vision/VisionTypes.h"

#include <vector>


namespace Saiga
{
struct SAIGA_GLOBAL Extrinsics
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sophus::SE3d se3;
    bool constant = false;

    Eigen::Vector3d apply(const Eigen::Vector3d& X) { return se3 * X; }
};

struct SAIGA_GLOBAL WorldPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d p;
    bool valid = false;

    // Pair < ImageID, ImagePointID >
    std::vector<std::pair<int, int> > references;

    bool isReferencedByFrame(int i)
    {
        for (auto p : references)
            if (p.first == i) return true;
        return false;
    }

    void removeReference(int img, int id)
    {
        SAIGA_ASSERT(isReferencedByFrame(img));
        references.erase(std::find(references.begin(), references.end(), std::make_pair(img, id)));
    }

    bool isValid() { return valid && !references.empty(); }
};

struct SAIGA_GLOBAL MonoImagePoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int wp = -1;

    Eigen::Vector2d point;
    float weight = 1;


    // === computed by reprojection
    double repDepth = 0;
    Eigen::Vector2d repPoint;
};

struct SAIGA_GLOBAL StereoImagePoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int wp = -1;

    double depth = 0;
    Eigen::Vector2d point;
    float weight = 1;


    // === computed by reprojection
    double repDepth = 0;
    Eigen::Vector2d repPoint;
};

struct SAIGA_GLOBAL DenseConstraint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool print = false;
    Eigen::Vector2d initialProjection;
    // The depth + the (fixed) extrinsics of the reference is enough to project this point to the
    // target frame. Then we can compute the photometric and geometric error with the projected depth
    // and the reference intensity
    double referenceDepth;
    double referenceIntensity;

    Eigen::Vector2d referencePoint;

    int targetImageId = 0;
    float weight      = 1;

    // reference point projected to world space
    // (only works if the reference se3 is fixed)
    Eigen::Vector3d referenceWorldPoint;
};

struct SAIGA_GLOBAL SceneImage
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int intr = -1;
    int extr = -1;
    std::vector<MonoImagePoint> monoPoints;
    std::vector<StereoImagePoint> stereoPoints;
    std::vector<DenseConstraint> densePoints;
    float imageWeight = 1;

    float imageScale = 1;
    Saiga::TemplatedImage<float> intensity;
    Saiga::TemplatedImage<float> gIx, gIy;

    Saiga::TemplatedImage<float> depth;
    Saiga::TemplatedImage<float> gDx, gDy;
};


class SAIGA_GLOBAL Scene
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<Intrinsics4> intrinsics;
    std::vector<Extrinsics> extrinsics;
    std::vector<WorldPoint> worldPoints;

    double bf;

    std::vector<SceneImage> images;

    Vec3 residual(const SceneImage& img, const StereoImagePoint& ip);
    Vec2 residual(const SceneImage& img, const MonoImagePoint& ip);

    // Apply a rigid transformation to the complete scene
    void transformScene(const SE3& transform);

    void fixWorldPointReferences();

    bool valid();

    double rms();
    double rmsDense();

    // Computes the median point from all valid world points
    Vec3 medianWorldPoint();

    // remove all image points which project to negative depth values (behind the camera)
    void removeNegativeProjections();

    Saiga::Statistics<double> statistics();
    void removeOutliers(float factor);
};

}  // namespace Saiga
