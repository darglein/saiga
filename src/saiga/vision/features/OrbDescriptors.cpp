
#include "OrbDescriptors.h"

#include "saiga/core/time/all.h"
#include "saiga/core/util/Thread/omp.h"

#include "OrbPattern.h"

#include <vector>
using namespace std;

namespace Saiga
{
// const int PATCH_SIZE      = 31;
const int HALF_PATCH_SIZE = 15;
// const int EDGE_THRESHOLD  = 19;


ORB::ORB()
{
    u_max = ORBPattern::AngleUmax();
    descriptor_pattern =
        std::vector<ivec2>(ORBPattern::DescriptorPattern().begin(), ORBPattern::DescriptorPattern().end());
}

float ORB::ComputeAngle(Saiga::ImageView<unsigned char> image, const Saiga::vec2& pt)
{
    int m_01 = 0, m_10 = 0;


    //    const uchar* center = &image.at<uchar>(cvRound(pt.y()), cvRound(pt.x()));
    const unsigned char* center = &image(Saiga::iRound(pt.y()), Saiga::iRound(pt.x()));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u) m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.pitchBytes;
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d     = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
    float angle = Saiga::degrees(atan2((float)m_01, (float)m_10));
    return (angle < 0) * 360 + angle;
}



DescriptorORB ORB::ComputeDescriptor(Saiga::ImageView<unsigned char> image, const vec2& point, float angle_degrees)
{
    DescriptorORB result;
    auto desc = (unsigned char*)&result;

    const ivec2* pattern = descriptor_pattern.data();


    float angle = Saiga::radians(angle_degrees);
    float a = (float)cos(angle), b = (float)sin(angle);

#if 0
    const unsigned char* center = &image(iRound(point.y()), iRound(point.x()));
    const int step              = (int)image.pitchBytes;
#endif


    auto GET_VALUE = [&](int idx) -> int {

#if 1
        float fx = point.x() + (pattern[idx].x() * a - pattern[idx].y() * b);
        float fy = point.y() + (pattern[idx].x() * b + pattern[idx].y() * a);
        int x    = iRound(fx);
        int y    = iRound(fy);

        //        SAIGA_ASSERT(image.inImage(y, x));
        //        image.clampToEdge(y, x);
        return image(y, x);
#else
        return center[cvRound(pattern[idx].x() * b + pattern[idx].y() * a) * step +
                      cvRound(pattern[idx].x() * a - pattern[idx].y() * b)];
#endif
    };


    //#pragma unroll
    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0  = GET_VALUE(0);
        t1  = GET_VALUE(1);
        val = t0 < t1;
        t0  = GET_VALUE(2);
        t1  = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (unsigned char)val;
    }
    return result;
}
}  // namespace Saiga
