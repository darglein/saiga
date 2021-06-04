/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/VisionTypes.h"

#include "Triangulation.h"

#include <array>

namespace Saiga
{
/**
 * Computes the essential matrix from given camera extrinsics.
 * The resulting essential matrix maps points of (a) to lines in (b).
 *
 * 1. Computes relative rotation and translation [R|t] between cameras
 * 2. The essential matrix E is then: E = R[t]x
 *
 *
 * Hartley - Chapter 9.6 Essential Matrix (page 257):
 *
 * Given two normalized cameras P, P' with
 * P  = [I | 0]
 * P' = [R | t]
 * then the essential matrix mapping a point x in P to a line l in P' is
 * E = [t]xR
 *
 * If P,P' are not normalized we multiply P^-1 right sided to both matrices.
 * P2  = P  * P^-1 = [I|0]
 * P2' = P2 * P^-1 = [R|t]
 *
 * Relation to fundamental Matrix:
 * E = K'^T*F*K
 * F = K'^T^-1 * E * K^-1
 *
 */
SAIGA_VISION_API Mat3 EssentialMatrix(const SE3& a, const SE3& b);
SAIGA_VISION_API Mat3 EssentialMatrix(const Mat3& F, const IntrinsicsPinholed& K1, const IntrinsicsPinholed& K2);


/**
 * Computes the Fundamental Matrix given an essential matrix and
 * the camera intrinsics.
 * Assumes a pinhole camera model!
 *
 * F = K'^T^-1 * E * K^-1
 * F = K2^-T * E * K1^-1
 */
SAIGA_VISION_API Mat3 FundamentalMatrix(const Mat3& E, const IntrinsicsPinholed& K1, const IntrinsicsPinholed& K2);


// Normalize essential or fundamental matrix.
SAIGA_VISION_API Mat3 NormalizeEpipolarMatrix(const Mat3& EorF);



/**
 * Computes the squared distance of point 2 to the epipolar line of point 1.
 */
SAIGA_VISION_API double EpipolarDistanceSquared(const Vec2& p1, const Vec2& p2, const Mat3& F);



// estimate the rotation and translation of the camera given the essential matrix E
// see:
// Richard Hartley and Andrew Zisserman (2003). Multiple View Geometry in computer vision
// http://isit.u-clermont1.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf
SAIGA_VISION_API void decomposeEssentialMatrix(const Mat3& E, Mat3& R1, Mat3& R2, Vec3& t1, Vec3& t2);

SAIGA_VISION_API std::array<SE3, 4> decomposeEssentialMatrix2(Mat3& E);

SAIGA_VISION_API std::pair<SE3, int> getValidTransformationFromE(Mat3& E, const Vec2* points1, const Vec2* points2,
                                                                 int N);

SAIGA_VISION_API std::pair<SE3, int> getValidTransformationFromE(Mat3& E, const Vec2* points1, const Vec2* points2,
                                                                 ArrayView<char> inlier_mask, int N, int num_threads);

}  // namespace Saiga
