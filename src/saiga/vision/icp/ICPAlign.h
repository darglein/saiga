/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <vector>
#include "saiga/vision/VisionTypes.h"

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

namespace Saiga {
namespace ICP {



struct SAIGA_GLOBAL Correspondence
{
    Vec3 refPoint;
    Vec3 refNormal;
    Vec3 srcPoint;
    Vec3 srcNormal;
    double weight = 1;

    // Apply this transfomration to the src point and normal
    void apply(const SE3& se3)
    {
        srcPoint = se3 * srcPoint;
        srcNormal = se3.so3() * srcNormal;
    }
};

/**
 * The basic ICP algorithm which minimized the function above. Each correspondence only needs
 * the 'refPoint' and 'srcPoint'.
 */
SAIGA_GLOBAL SE3 pointToPoint(const std::vector<Correspondence>& corrs, const SE3& guess = SE3());

/**
 * Minimized the distance between the source point to the surface plane at the reference point:
 *
 * argmin_T || (ref_i - T*src_i)*ref_normal_i ||^2
 *
 * Each correspondnce additional needs the 'refNormal' attribute.
 */
SAIGA_GLOBAL SE3 pointToPlane(const std::vector<Correspondence>& corrs, const SE3& guess = SE3(), int innerIterations = 1);


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
SAIGA_GLOBAL SE3 planeToPlane(const std::vector<Correspondence>& corrs, const SE3& guess = SE3(), double covE = 0.001, int innerIterations = 5);


}
}