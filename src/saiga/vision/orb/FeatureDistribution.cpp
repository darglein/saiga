#include "FeatureDistribution.h"

#include <algorithm>
#include <vector>
#include "Nanoflann.h"
#include "Rangetree.h"
#include "PointCloud.h"
#include "Quadtree.h"

namespace Saiga
{
int FeatureDistributionTopN::operator()(std::vector<kpt_t>& keypoints)
{
    if (keypoints.size() <= N)
        return keypoints.size();
    std::nth_element(keypoints.begin(), keypoints.begin()+N, keypoints.end(),
                     [](const kpt_t& k1, const kpt_t& k2){return k1.response > k2.response;});
    keypoints.resize(N);
    return keypoints.size();
}

int FeatureDistributionBucketing::operator()(std::vector<kpt_t>& keypoints)
{
    if (keypoints.size() <= N)
        return keypoints.size();

    int width = imageSize[0];
    int height = imageSize[1];
    int cellSizeX = std::min(bucketSize[0], width);
    int cellSizeY = std::min(bucketSize[1], height);

    const int npatchesInX = width / cellSizeX;
    const int npatchesInY = height / cellSizeY;
    const int patchWidth = ceil((double)width / npatchesInX);
    const int patchHeight = ceil((double)height / npatchesInY);

    int nCells = npatchesInX * npatchesInY;
    std::vector<std::vector<kpt_t>> cellkpts(nCells);
    int nPerCell = (float)N / nCells;
    if (nPerCell < 1)
        nPerCell = 1;

    for (auto &kpt : keypoints)
    {
        int idx = (int)(kpt.point.y()/patchHeight) * npatchesInX + (int)(kpt.point.x()/patchWidth);
        if (idx >= nCells)
            idx = nCells-1;
        cellkpts[idx].emplace_back(kpt);
    }

    keypoints.clear();

    for (auto &kptVec : cellkpts)
    {
        if (kptVec.size() > nPerCell)
        {
            std::nth_element(kptVec.begin(), kptVec.begin()+nPerCell, kptVec.end(),
                    [](const kpt_t& k1, const kpt_t& k2){return k1.response > k2.response;});
            kptVec.resize(nPerCell);
        }
        keypoints.insert(keypoints.end(), kptVec.begin(), kptVec.end());
    }
    return keypoints.size();
}

int FeatureDistributionQuadtree::operator()(std::vector<kpt_t> &keypoints)
{
    if (keypoints.size() <= N)
        return keypoints.size();
    const int nroots = round(static_cast<float>(imageSize[0])/(imageSize[1]));

    const float nodeWidth = static_cast<float>(imageSize[0]) / nroots;

    std::list<QuadtreeNode> nodesList;

    std::vector<QuadtreeNode*> rootVec;
    rootVec.resize(nroots);


    for (int i = 0; i < nroots; ++i)
    {
        int x0 = nodeWidth * (float)i;
        int x1 = nodeWidth * (float)(i+1);
        int y0 = 0;
        int y1 = imageSize[1];
        QuadtreeNode n;
        n.UL = ivec2(x0, y0);
        n.UR = ivec2(x1, y0);
        n.LL = ivec2(x0, y1);
        n.LR = ivec2(x1, y1);
        n.nodeKpts.reserve(keypoints.size());

        nodesList.push_back(n);
        rootVec[i] = &nodesList.back();
    }


    for (auto &kpt : keypoints)
    {
        rootVec[(int)(kpt.point.x() / nodeWidth)]->nodeKpts.emplace_back(kpt);
    }

    std::list<QuadtreeNode>::iterator current;
    current = nodesList.begin();

    while (current != nodesList.end())
    {
        if (current->nodeKpts.size() == 1)
        {
            current->leaf = true;
            ++current;
        }
        else if (current->nodeKpts.empty())
        {
            current = nodesList.erase(current);
        }
        else
            ++current;
    }

    std::vector<QuadtreeNode*> nodesToExpand;
    nodesToExpand.reserve(nodesList.size()*4);

    bool omegadoom = false;
    int lastSize = 0;
    while (!omegadoom)
    {
        current = nodesList.begin();
        lastSize = nodesList.size();

        nodesToExpand.clear();
        int nToExpand = 0;

        while (current != nodesList.end())
        {
            if (current->leaf)
            {
                ++current;
                continue;
            }

            QuadtreeNode n1, n2, n3, n4;
            current->DivideNode(n1, n2, n3, n4);
            if (!n1.nodeKpts.empty())
            {
                nodesList.push_front(n1);
                if (n1.nodeKpts.size() == 1)
                    n1.leaf = true;
                else
                {
                    ++nToExpand;
                    nodesToExpand.emplace_back(&nodesList.front());
                    nodesList.front().lit = nodesList.begin();
                }
            }
            if (!n2.nodeKpts.empty())
            {
                nodesList.push_front(n2);
                if (n2.nodeKpts.size() == 1)
                    n2.leaf = true;
                else
                {
                    ++nToExpand;
                    nodesToExpand.emplace_back(&nodesList.front());
                    nodesList.front().lit = nodesList.begin();
                }
            }
            if (!n3.nodeKpts.empty())
            {
                nodesList.push_front(n3);
                if (n3.nodeKpts.size() == 1)
                    n3.leaf = true;
                else
                {
                    ++nToExpand;
                    nodesToExpand.emplace_back(&nodesList.front());
                    nodesList.front().lit = nodesList.begin();
                }
            }
            if (!n4.nodeKpts.empty())
            {
                nodesList.push_front(n4);
                if (n4.nodeKpts.size() == 1)
                    n4.leaf = true;
                else
                {
                    ++nToExpand;
                    nodesToExpand.emplace_back(&nodesList.front());
                    nodesList.front().lit = nodesList.begin();
                }
            }

            current = nodesList.erase(current);

        }
        if ((int)nodesList.size() >= N || (int)nodesList.size()==lastSize)
        {
            omegadoom = true;
        }

        else if ((int)nodesList.size() + nToExpand*3 > N)
        {
            while(!omegadoom)
            {
                lastSize = nodesList.size();
                std::vector<QuadtreeNode*> prevNodes = nodesToExpand;

                nodesToExpand.clear();

                std::sort(prevNodes.begin(), prevNodes.end(),
                          [](const QuadtreeNode *n1, const QuadtreeNode *n2)
                          {return n1->nodeKpts.size() > n2->nodeKpts.size();});

                for (auto &node : prevNodes)
                {
                    QuadtreeNode n1, n2, n3, n4;
                    node->DivideNode(n1, n2, n3, n4);

                    if (!n1.nodeKpts.empty())
                    {
                        nodesList.push_front(n1);
                        if (n1.nodeKpts.size() == 1)
                            n1.leaf = true;
                        else
                        {
                            nodesToExpand.emplace_back(&nodesList.front());
                            nodesList.front().lit = nodesList.begin();
                        }

                    }
                    if (!n2.nodeKpts.empty())
                    {
                        nodesList.push_front(n2);
                        if (n2.nodeKpts.size() == 1)
                            n2.leaf = true;
                        else
                        {
                            nodesToExpand.emplace_back(&nodesList.front());
                            nodesList.front().lit = nodesList.begin();
                        }

                    }
                    if (!n3.nodeKpts.empty())
                    {
                        nodesList.push_front(n3);
                        if (n3.nodeKpts.size() == 1)
                            n3.leaf = true;
                        else
                        {
                            nodesToExpand.emplace_back(&nodesList.front());
                            nodesList.front().lit = nodesList.begin();
                        }

                    }
                    if (!n4.nodeKpts.empty())
                    {
                        nodesList.push_front(n4);
                        if (n4.nodeKpts.size() == 1)
                            n4.leaf = true;
                        else
                        {
                            nodesToExpand.emplace_back(&nodesList.front());
                            nodesList.front().lit = nodesList.begin();
                        }

                    }
                    nodesList.erase(node->lit);

                    if ((int)nodesList.size() >= N)
                        break;
                }
                if ((int)nodesList.size() >= N || (int)nodesList.size() == lastSize)
                    omegadoom = true;


            }
        }
    }


    std::vector<kpt_t> resKpts;
    resKpts.reserve(N*2);
    auto iter = nodesList.begin();
    for (; iter != nodesList.end(); ++iter)
    {
        std::vector<kpt_t> &nodekpts = iter->nodeKpts;
        kpt_t* kpt = &nodekpts[0];
        if (iter->leaf)
        {
            resKpts.emplace_back(*kpt);
            continue;
        }

        float maxScore = kpt->response;
        for (auto &k : nodekpts)
        {
            if (k.response > maxScore)
            {
                kpt = &k;
                maxScore = k.response;
            }

        }
        resKpts.emplace_back(*kpt);
    }

    keypoints = resKpts;
    return keypoints.size();
}

int FeatureDistributionANMS::operator()(std::vector<kpt_t> &keypoints)
{
    if (keypoints.size() <= N)
        return keypoints.size();

    int cols = imageSize[0], rows = imageSize[1];
    int numerator1 = rows + cols + 2*N;
    long long discriminant = (long long)4*cols + (long long)4*N + (long long)4*rows*N +
                             (long long)rows*rows + (long long)cols*cols - (long long)2*cols*rows + (long long)4*cols*rows*N;

    double denominator = 2*(N-1);

    double sol1 = (numerator1 - sqrt(discriminant))/denominator;
    double sol2 = (numerator1 + sqrt(discriminant))/denominator;

    int high = (sol1>sol2)? sol1 : sol2; //binary search range initialization with positive solution
    int low = std::floor(sqrt((double)keypoints.size()/N));

    switch (ac)
    {
        case AccelerationStructure::KDTREE:
        {
            ANMSKDTree(keypoints, high, low);
            return keypoints.size();
        }
        case AccelerationStructure::RANGETREE:
        {
            ANMSRangeTree(keypoints, high, low);
            return keypoints.size();
        }
        case AccelerationStructure::GRID:
        {
            ANMSGrid(keypoints, high, low);
            return keypoints.size();
        }
        default:
        {
            return 0;
        }
    }
}

int FeatureDistributionANMS::ANMSKDTree(std::vector<kpt_t> &keypoints, int high, int low)
{
    KdTreePointCloud<int> cloud;
    generatePointCloud(cloud, keypoints);
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<int, KdTreePointCloud<int>>,
            KdTreePointCloud<int>, 2> a_kd_tree;
    a_kd_tree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(25));
    tree.buildIndex();

    bool done = false;
    int kMax = round(N + N*epsilon), kMin = round(N - N*epsilon);
    std::vector<int> resultIndices;
    int radius, prevradius = 1;

    std::vector<int> tempResult;
    tempResult.reserve(keypoints.size());
    while (!done)
    {
        std::vector<bool> selected(keypoints.size(), true);
        radius = low + (high-low)/2;
        if (radius == prevradius || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();

        for (int i = 0; i < keypoints.size(); ++i)
        {
            if (selected[i])
            {
                selected[i] = false;
                tempResult.emplace_back(i);
                int searchRadius = static_cast<int>(radius*radius);
                std::vector<std::pair<size_t, int>> retMatches;
                nanoflann::SearchParams params;
                int querypt[2] = {(int)keypoints[i].point.x(), (int)keypoints[i].point.y()};
                size_t nMatches = tree.radiusSearch(&querypt[0], searchRadius, retMatches, params);

                for (size_t idx = 0; idx < nMatches; ++idx)
                {
                    if(selected[retMatches[idx].first])
                        selected[retMatches[idx].first] = false;
                }
            }
        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = radius - 1;
        else
            low = radius + 1;

        prevradius = radius;
    }
    std::vector<kpt_t> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(keypoints[resultIndices[i]]);
    }
    keypoints = reskpts;
    return keypoints.size();
}

int FeatureDistributionANMS::ANMSRangeTree(std::vector<kpt_t> &keypoints, int high, int low)
{
    RangeTree<u16, u16> tree(keypoints.size(), keypoints.size());
    for (int i = 0; i < keypoints.size(); ++i)
    {
        tree.add(keypoints[i].point.x(), keypoints[i].point.y(), (u16 *)(intptr_t)i);
    }
    tree.finalize();

    bool done = false;
    int kMin = round(N - N*epsilon), kMax = round(N + N*epsilon);
    std::vector<int> resultIndices;
    int width, prevwidth = -1;

    std::vector<int> tempResult;
    tempResult.reserve(keypoints.size());

    while (!done)
    {
        std::vector<bool> selected(keypoints.size(), true);
        width = low + (high-low)/2;
        if (width == prevwidth || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();

        for (int i = 0; i < keypoints.size(); ++i)
        {
            if (selected[i])
            {
                selected[i] = false;
                tempResult.emplace_back(i);
                int minX = static_cast<int>(keypoints[i].point.x() - width);
                int maxX = static_cast<int>(keypoints[i].point.x() + width);
                int minY = static_cast<int>(keypoints[i].point.y() - width);
                int maxY = static_cast<int>(keypoints[i].point.y() + width);

                if (minX < 0)
                    minX = 0;
                if (minY < 0)
                    minY = 0;


                std::vector<u16*> *he = tree.search(minX, maxX, minY, maxY);
                for (int j = 0; j < he->size(); ++j)
                {
                    if (selected[(u64)(*he)[j]])
                        selected[(u64)(*he)[j]] = false;
                }
                delete he;
                he = nullptr;
            }

        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = width - 1;
        else
            low = width + 1;

        prevwidth = width;
    }

    std::vector<kpt_t> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(keypoints[resultIndices[i]]);
    }
    keypoints = reskpts;
    return keypoints.size();
}

int FeatureDistributionANMS::ANMSGrid(std::vector<kpt_t> &keypoints, int high, int low)
{
    bool done = false;
    int kMin = round(N - N*epsilon), kMax = round(N + N*epsilon);
    std::vector<int> resultIndices;
    int width, prevwidth = -1;

    std::vector<int> tempResult;
    tempResult.reserve(keypoints.size());

    while(!done)
    {
        width = low + (high-low)/2;
        if (width == prevwidth || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();
        double c = (double)width/2.0;
        int cellCols = std::floor(imageSize[0]/c);
        int cellRows = std::floor(imageSize[1]/c);
        std::vector<std::vector<bool>> covered(cellRows+1, std::vector<bool>(cellCols+1, false));

        for (int i = 0; i < keypoints.size(); ++i)
        {
            int row = (int)(keypoints[i].point.y()/c);
            int col = (int)(keypoints[i].point.x()/c);

            if (covered[row][col] == false)
            {
                tempResult.emplace_back(i);
                int rowMin = std::max(0, row - 2);
                int rowMax = std::min(cellRows, row + 2);
                int colMin = std::max(0, col - 2);
                int colMax = std::min (cellCols, col + 2);

                for (int dy = rowMin; dy <= rowMax; ++dy)
                {
                    for (int dx = colMin; dx <= colMax; ++dx)
                    {
                        if (!covered[dy][dx])
                            covered[dy][dx] = true;
                    }
                }
            }
        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = width - 1;
        else
            low = width + 1;

        prevwidth = width;
    }

    std::vector<kpt_t> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(keypoints[resultIndices[i]]);
    }
    keypoints = reskpts;
    return keypoints.size();
}

int FeatureDistributionSoftSSC::operator()(std::vector<kpt_t> &keypoints)
{
    if (keypoints.size() <= N)
        return keypoints.size();

    int cols = imageSize[0], rows = imageSize[1];
    int numerator1 = rows + cols + 2*N;
    long long discriminant = (long long)4*cols + (long long)4*N + (long long)4*rows*N +
                             (long long)rows*rows + (long long)cols*cols - (long long)2*cols*rows + (long long)4*cols*rows*N;

    double denominator = 2*(N-1);

    double sol1 = (numerator1 - sqrt(discriminant))/denominator;
    double sol2 = (numerator1 + sqrt(discriminant))/denominator;

    int high = (sol1>sol2)? sol1 : sol2; //binary search range initialization with positive solution
    int low = std::floor(sqrt((double)keypoints.size()/N));

    bool done = false;
    int kMin = round(N - N*epsilon), kMax = round(N + N*epsilon);
    std::vector<int> resultIndices;
    int width, prevwidth = -1;

    std::vector<int> tempResult;
    tempResult.reserve(keypoints.size());

    while(!done)
    {
        width = low + (high-low)/2;
        if (width == prevwidth || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();
        double c = (double)width/2.0;
        int cellCols = std::floor(imageSize[0]/c);
        int cellRows = std::floor(imageSize[1]/c);
        std::vector<std::vector<float>> covered(cellRows+1, std::vector<float>(cellCols+1, -1));

        for (int i = 0; i < keypoints.size(); ++i)
        {
            int row = (int)(keypoints[i].point.y()/c);
            int col = (int)(keypoints[i].point.x()/c);

            float score = keypoints[i].response;

            if (covered[row][col] < score + threshold)
            {
                tempResult.emplace_back(i);
                int rowMin = std::max(0, row - 2);
                int rowMax = std::min(cellRows, row + 2);
                int colMin = std::max(0, col - 2);
                int colMax = std::min (cellCols, col + 2);

                for (int dy = rowMin; dy <= rowMax; ++dy)
                {
                    for (int dx = colMin; dx <= colMax; ++dx)
                    {
                        if (covered[dy][dx] < score)
                            covered[dy][dx] = score;
                    }
                }
            }
        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = width - 1;
        else
            low = width + 1;

        prevwidth = width;
    }

    std::vector<kpt_t> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(keypoints[resultIndices[i]]);
    }
    keypoints = reskpts;
    return keypoints.size();
}
}  // namespace Saiga
