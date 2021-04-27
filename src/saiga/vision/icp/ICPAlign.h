/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

#include <vector>

/**
 * Iterative Closest Point (ICP) Algorithm for aligning two point clouds.
 *
 * Given a reference (ref) point cloud and a source (src) point cloud the goal is to find
 * a rigid body transformation T that maps src on dst.
 *
 * The pairwise point matches are already given and can be, for example, computed with
 * a nearest neighbour search or projective mapping. Each point in src_i should be
 * mapped with T on a reference point ref_i.
 *
 * ref_i = T * src_i
 * argmin_T || ref_i - T * src_i ||^2
 *
 * Each correspondence includes a weight, to allow the user to downweight some matches.
 * The rotation is optimized in the gauss-newton steps in the tagent space of SO3
 * with the lie algebra and matrix exponentials.
 * If more information for each point is given (for example the local normal), we can use
 * better distance metrics for better results. See the functions below for more information.
 *
 */

namespace Saiga
{
namespace ICP
{
struct SAIGA_VISION_API Correspondence
{
    Vec3 refPoint;
    Vec3 refNormal;
    Vec3 srcPoint;
    Vec3 srcNormal;
    double weight = 1;

    // Apply this transfomration to the src point and normal
    void apply(const SE3& T)
    {
        srcPoint  = T * srcPoint;
        srcNormal = T.so3() * srcNormal;
    }



    double residualPointToPoint() { return (refPoint - srcPoint).squaredNorm(); }

    double residualPointToPlane()
    {
        double d = refNormal.dot(refPoint - srcPoint);
        return d * d;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * The basic point to point registration algorithm which minimized the function above. Each correspondence only needs
 * the 'refPoint' and 'srcPoint'. The function is minized iterativley using the Gauss-Newton algorithm.
 */
SAIGA_VISION_API SE3 pointToPointIterative(const AlignedVector<Correspondence>& corrs, const SE3& guess = SE3(),
                                           int innerIterations = 5);

/**
 * Analytical solution to the point cloud registration problem.
 * The funtion is solved using the polar decomposition.
 * See also "Orthonormal Procrustus Problem".
 *
 * If scale != nullptr a scaling between the point clouds is also computed
 *
 *
 */
SAIGA_VISION_API SE3 pointToPointDirect(const AlignedVector<Correspondence>& corrs, double* scale = nullptr);

/**
 * Minimized the distance between the source point to the surface plane at the reference point:
 *
 * argmin_T || (ref_i - T*src_i)*ref_normal_i ||^2
 *
 * Each correspondnce additional needs the 'refNormal' attribute.
 */
SAIGA_VISION_API SE3 pointToPlane(const AlignedVector<Correspondence>& corrs, const SE3& ref, const SE3& src,
                                  int innerIterations = 1);


/**
 * Minimized a symmetric diffrence between both points and both surfaces:
 *
 * argmin_T  (ref_i-T*src_i)^T * (C_ref + T_r*C_src*T_r^T)^-1 * (ref_i-T*src_i)^T,
 *
 * where C_ref, C_src are the covariance matrices at the two points. To solve this non-linear problem,
 * we apply the same approximation as G2O:
 *
 * In each iteration the matrix in the middle (including the covariance matrices) is kept constant.
 * Then this function is minimized by a single Gauss-Newton step.
 *
 * Full Paper: Generalized-ICP, http://www.roboticsproceedings.org/rss05/p21.pdf
 * G2O Implementation: https://github.com/RainerKuemmerle/g2o/blob/master/g2o/types/icp/types_icp.h
 */
SAIGA_VISION_API SE3 planeToPlane(const AlignedVector<Correspondence>& corrs, const SE3& guess = SE3(),
                                  double covE = 0.001, int innerIterations = 5);


SAIGA_VISION_API Quat orientationFromMixedMatrixUQ(const Mat3& M);
SAIGA_VISION_API Quat orientationFromMixedMatrixSVD(const Mat3& M);

// aligning 3 points from src to dst.
// this is the minimal problem and used for example in the p3p solution.
// similar to eigen::umeyama but faster
SAIGA_VISION_API SE3 alignMinimal(const Mat3& src, const Mat3& dst);
}  // namespace ICP
}  // namespace Saiga
