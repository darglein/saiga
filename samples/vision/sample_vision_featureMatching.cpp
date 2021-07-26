/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/image/image.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/reconstruction/EightPoint.h"
#include "saiga/vision/features/Features.h"

#include <numeric>
using namespace Saiga;

using FeatureDescriptor = DescriptorORB;

std::vector<DescriptorORB> randomDescriptors(int n)
{
    std::vector<DescriptorORB> des(n);
    for (auto& d : des)
    {
        for (auto& dd : d) dd = Random::urand64();
    }
    return des;
}

int main(int, char**)
{
    initSaigaSampleNoWindow();

    Saiga::EigenHelper::checkEigenCompabitilty<15357>();
    Saiga::Random::setSeed(45786045);



    std::vector<KeyPoint<double>> keys1, keys2;
    std::vector<FeatureDescriptor> des1, des2;

    if (0)
    {
        des1 = randomDescriptors(1500);
        des2 = randomDescriptors(1000);
    }
    else
    {
        auto p1 = SearchPathes::data("vision/0.features");
        auto p2 = SearchPathes::data("vision/10.features");
        SAIGA_ASSERT(!p1.empty() && !p2.empty());

        {
            BinaryFile bf(p1, std::ios_base::in);
            bf >> keys1 >> des1;
        }
        {
            BinaryFile bf(p2, std::ios_base::in);
            bf >> keys2 >> des2;
        }

        //        BinaryFile bf(tmpDir + "/" + to_string(image.frameId) + ".features", std::ios_base::in);
        //        SAIGA_ASSERT(bf.strm.is_open());
        //        bf >> features.mvKeys >> features.descriptors;
    }


    int n = des1.size();
    int m = des2.size();

    std::vector<int> d_matchMatrix(n * m);
    ImageView<int> matchMatrix(n, m, d_matchMatrix.data());

    std::cout << "Bruteforce matching " << n << "x" << m << " ORB Descriptors..." << std::endl;
    {
        SAIGA_BLOCK_TIMER("Matching");
        for (auto i : Range(0, n))
        {
            for (auto j : Range(0, m))
            {
                matchMatrix(i, j) = distance(des1[i], des2[j]);
            }
        }
    }
    auto sum = std::accumulate(d_matchMatrix.begin(), d_matchMatrix.end(), 0);
    std::cout << "distance sum: " << sum << " avg: " << double(sum) / (n * m) << std::endl;


    BruteForceMatcher<DescriptorORB> matcher;
    matcher.match(des1.begin(), n, des2.begin(), m);
    matcher.matchKnn2(des1, des2);
    matcher.filterMatches(50, 0.6);
    return 0;
}
