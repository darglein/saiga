#pragma once

#include <vector>
#include "Types.h"

namespace Saiga
{
template <typename T>
struct KdTreePointCloud
{
std::vector<Eigen::Matrix<T,2,1>>  pts;
inline size_t kdtree_get_point_count() const { return pts.size(); }
inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t /*size*/) const
{
    const T d0=p1[0]-pts[idx_p2].x();
    const T d1=p1[1]-pts[idx_p2].y();
    return d0*d0+d1*d1;
}

inline T kdtree_get_pt(const size_t idx, int dim) const
{
    if (dim==0) return pts[idx].x();
    else if (dim==1) return pts[idx].y();
    return 0;
}

template <class BBOX>
bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

template <typename T>
void generatePointCloud(KdTreePointCloud<T> &point, std::vector<Saiga::KeyPoint<float>> keypoints)
{
    point.pts.resize(keypoints.size());
    for (size_t i=0;i<keypoints.size();i++)
    {
        point.pts[i].x() = keypoints[i].point.x();
        point.pts[i].y() = keypoints[i].point.y();
    }
}

} //namespace Saiga