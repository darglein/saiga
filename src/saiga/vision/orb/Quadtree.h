#pragma once

#include "Types.h"
#include <list>

namespace Saiga
{

class QuadtreeNode
{
public:
    QuadtreeNode():leaf(false){}

    void DivideNode(QuadtreeNode &n1, QuadtreeNode &n2, QuadtreeNode &n3, QuadtreeNode &n4);

    std::vector<kpt_t> nodeKpts;
    ivec2 UL, UR, LL, LR;
    std::list<QuadtreeNode>::iterator lit;
    bool leaf;
};

void QuadtreeNode::DivideNode(QuadtreeNode &n1, QuadtreeNode &n2, QuadtreeNode &n3, QuadtreeNode &n4)
{
    int middleX = UL.x() + (int)std::ceil((float)(UR.x() - UL.x())/2.f);
    int middleY = UL.y() + (int)std::ceil((float)(LL.y() - UL.y())/2.f);

    ivec2 M (middleX, middleY);
    ivec2 upperM (middleX, UL.y());
    ivec2 lowerM (middleX, LL.y());
    ivec2 leftM (UL.x(), middleY);
    ivec2 rightM (UR.x(), middleY);

    n1.UL = UL, n1.UR = upperM, n1.LL = leftM, n1.LR = M;
    n2.UL = upperM, n2.UR = UR, n2.LL = M, n2.LR = rightM;
    n3.UL = leftM, n3.UR = M, n3.LL = LL, n3.LR = lowerM;
    n4.UL = M, n4.UR = rightM, n4.LL = lowerM, n4.LR = LR;

    for (auto &kpt : nodeKpts)
    {
        if (kpt.point.x()< middleX)
        {
            if(kpt.point.y() < middleY)
                n1.nodeKpts.emplace_back(kpt);
            else
                n3.nodeKpts.emplace_back(kpt);

        }
        else
        {
            if (kpt.point.y() < middleY)
                n2.nodeKpts.emplace_back(kpt);
            else
                n4.nodeKpts.emplace_back(kpt);
        }
    }
}

} // namespace Saiga