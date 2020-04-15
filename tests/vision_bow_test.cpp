/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/core/time/all.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/slam/MiniBow.h"
#include "saiga/vision/slam/MiniBow2.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
namespace Saiga
{
using Descriptor    = MiniBow::FORB::TDescriptor;
using OrbVocabulary = MiniBow::TemplatedVocabulary<Descriptor, MiniBow::FORB, MiniBow::L1Scoring>;

using Descriptor2    = MiniBow2::FORB::TDescriptor;
using OrbVocabulary2 = MiniBow2::TemplatedVocabulary<Descriptor, MiniBow2::FORB, MiniBow2::L1Scoring>;

const int images           = 3;
const int featuresPerImage = 1000;
void loadFeatures(std::vector<std::vector<Descriptor>>& features)
{
    features.clear();
    std::cout << "Computing random ORB features..." << std::endl;
    for (int i = 0; i < images; ++i)
    {
        std::vector<Descriptor> desc;
        for (auto j = 0; j < featuresPerImage; ++j)
        {
            Descriptor des;
            for (auto& d : des) d = Random::urand64();
            desc.push_back(des);
        }
        features.push_back(desc);
    }
}



void testVocMatching(const std::vector<std::vector<Descriptor>>& features, OrbVocabulary& voc)
{
    // lets do something with this vocabulary
    std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;

    double out = 0;

    std::vector<std::pair<MiniBow::BowVector, MiniBow::FeatureVector>> bows(features.size());
    for (int i = 0; i < features.size(); i++)
    {
        voc.transform(features[i], bows[i].first, bows[i].second, 4);
    }


    {
        for (int i = 0; i < features.size(); i++)
        {
            for (int j = 0; j < features.size(); j++)
            {
                double score = voc.score(bows[i].first, bows[j].first);
                out += score;
                std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
            }
        }
    }
    //    std::cout << "Score time: " << time / (features.size() * features.size()) << "ms" << std::endl;
}


void testVocMatching(const std::vector<std::vector<Descriptor>>& features, OrbVocabulary2& voc)
{
    // lets do something with this vocabulary
    std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;

    double out = 0;

    std::vector<std::pair<MiniBow2::BowVector, MiniBow2::FeatureVector>> bows(features.size());
    for (int i = 0; i < features.size(); i++)
    {
        voc.transform(features[i], bows[i].first, bows[i].second, 4);
    }


    {
        for (int i = 0; i < features.size(); i++)
        {
            for (int j = 0; j < features.size(); j++)
            {
                double score = voc.score(bows[i].first, bows[j].first);
                out += score;
                std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
            }
        }
    }
    //    std::cout << "Score time: " << time / (features.size() * features.size()) << "ms" << std::endl;
}


TEST(Sohpus, SE3_Point)
{
    OrbVocabulary orbVoc("ORBvoc.minibow");
    std::cout << orbVoc << std::endl;


    OrbVocabulary2 orbVoc2("ORBvoc.minibow");
    std::cout << orbVoc2 << std::endl;

    std::vector<std::vector<Descriptor>> features;
    loadFeatures(features);

    std::cout << "Testing Matching with ORB-SLAM Voc..." << std::endl;
    testVocMatching(features, orbVoc);
    testVocMatching(features, orbVoc2);
}

}  // namespace Saiga
