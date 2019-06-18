/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Trajectory.h"

#include "saiga/core/util/Range.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/icp/ICPAlign.h"

namespace Saiga
{
namespace Trajectory
{
double align(TrajectoryType& A, TrajectoryType& B)
{
    SAIGA_ASSERT(A.size() == B.size());
    if (A.empty()) return 0;
    int N = A.size();

    auto compFirst = [](const std::pair<int, SE3>& a, const std::pair<int, SE3>& b) { return a.first < b.first; };
    std::sort(A.begin(), A.end(), compFirst);
    std::sort(B.begin(), B.end(), compFirst);

    // transform both trajectories so that the first kf is at the origin
    SE3 pinv1 = A.front().second.inverse();
    SE3 pinv2 = B.front().second.inverse();
    for (auto& m : A) m.second = pinv1 * m.second;
    for (auto& m : B) m.second = pinv2 * m.second;


    // fit trajectories with icp
    AlignedVector<ICP::Correspondence> corrs;
    for (int i = 0; i < (int)A.size(); ++i)
    {
        ICP::Correspondence c;
        c.srcPoint = A[i].second.translation();
        c.refPoint = B[i].second.translation();
        corrs.push_back(c);
    }
    SE3 rel;
    rel = ICP::pointToPointDirect(corrs, rel, 4);

    // Apply transformation to the src trajectory (A)
    double error = 0;
    for (auto i : Range(0, N))
    {
        auto& c = corrs[i];
        error += c.residualPointToPoint();
        c.apply(rel);

        A[i].second = rel * A[i].second;
    }

    return error;
}

std::vector<double> rpe(const TrajectoryType& A, const TrajectoryType& B)
{
    SAIGA_ASSERT(A.size() == B.size());
    int N = A.size();
    std::vector<double> rpe;
    for (auto i : Range(1, N))
    {
        auto [a_id, a_se] = A[i];
        auto [b_id, b_se] = B[i];
        SAIGA_ASSERT(a_id == b_id);


        auto [a_id_prev, a_se_prev] = A[i - 1];
        auto [b_id_prev, b_se_prev] = B[i - 1];

        auto a_rel = a_se.inverse() * a_se_prev;
        auto b_rel = b_se.inverse() * b_se_prev;

        auto et              = translationalError(a_rel, b_rel);
        int numFramesBetween = a_id - a_id_prev;
        SAIGA_ASSERT(numFramesBetween > 0);
        et = et / double(numFramesBetween);
        rpe.push_back(et);
    }
    return rpe;
}

std::vector<double> ate(const TrajectoryType& A, const TrajectoryType& B)
{
    SAIGA_ASSERT(A.size() == B.size());
    int N = A.size();

    std::vector<double> ate;
    for (auto i : Range(0, N))
    {
        auto et = translationalError(A[i].second, B[i].second);
        ate.push_back(et);
    }
    return ate;
}

}  // namespace Trajectory
}  // namespace Saiga
