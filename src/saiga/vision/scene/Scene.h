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
    std::vector<std::pair<int, int> > stereoreferences;


    bool uniqueReferences() const
    {
        // check if all references are unique
        auto cpy = stereoreferences;
        std::sort(cpy.begin(), cpy.end());
        auto it = std::unique(cpy.begin(), cpy.end());
        return it == cpy.end();
    }

    bool isReferencedByStereoFrame(int i) const
    {
        for (auto p : stereoreferences)
            if (p.first == i) return true;
        return false;
    }



    void removeStereoReference(int img, int ip)
    {
        SAIGA_ASSERT(isReferencedByStereoFrame(img));
        for (auto& p : stereoreferences)
        {
            if (p.first == img && p.second == ip)
            {
                p = stereoreferences.back();
                break;
            }
        }
        stereoreferences.resize(stereoreferences.size() - 1);
        //        SAIGA_ASSERT(!isReferencedByStereoFrame(img));
    }

    // the valid flag is set and this point is referenced by at least one image
    bool isValid() const { return valid && (!stereoreferences.empty()); }

    explicit operator bool() const { return isValid(); }
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

    explicit operator bool() const { return wp != -1; }
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
    std::vector<StereoImagePoint> stereoPoints;
    std::vector<DenseConstraint> densePoints;
    float imageWeight = 1;

    float imageScale = 1;
    Saiga::TemplatedImage<float> intensity;
    Saiga::TemplatedImage<float> gIx, gIy;

    Saiga::TemplatedImage<float> depth;
    Saiga::TemplatedImage<float> gDx, gDy;

    int validPoints = 0;


    explicit operator bool() const { return valid(); }
    bool valid() const { return validPoints > 0; }
};


class SAIGA_GLOBAL Scene
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<Intrinsics4> intrinsics;
    std::vector<Extrinsics> extrinsics;
    std::vector<WorldPoint> worldPoints;

    // to scale towards [-1,1] range for floating point precision
    double globalScale = 1;

    double bf = 1;

    std::vector<SceneImage> images;

    double residualNorm2(const SceneImage& img, const StereoImagePoint& ip);
    Vec3 residual3(const SceneImage& img, const StereoImagePoint& ip);
    Vec2 residual2(const SceneImage& img, const StereoImagePoint& ip);
    double depth(const SceneImage& img, const StereoImagePoint& ip);

    // Apply a rigid transformation to the complete scene
    void transformScene(const SE3& transform);

    void fixWorldPointReferences();

    bool valid() const;
    explicit operator bool() const { return valid(); }

    double chi2();
    double rms();
    double rmsDense();

    double scale() { return globalScale; }

    // add 0-mean gaussian noise to the world points
    void addWorldPointNoise(double stddev);
    void addImagePointNoise(double stddev);
    void addExtrinsicNoise(double stddev);

    void sortByWorldPointId();

    // Computes the median point from all valid world points
    Vec3 medianWorldPoint();

    // remove all image points which project to negative depth values (behind the camera)
    void removeNegativeProjections();

    Saiga::Statistics<double> statistics();
    void removeOutliers(float factor);
    // removes all references to this worldpoint
    void removeWorldPoint(int id);
    void removeCamera(int id);

    // removes all worldpoints/imagepoints/images, which do not have any reference
    void compress();

    std::vector<int> validImages();
    std::vector<int> validPoints();

    // ================================= IO =================================
    // -> defined in Scene_io.cpp

    // returns true if the scene was changed by a user action
    bool imgui();
    void save(const std::string& file);
    void load(const std::string& file);
};

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& strm, Scene& scene);

}  // namespace Saiga
