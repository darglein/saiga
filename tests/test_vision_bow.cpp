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

using Descriptor2    = MiniBow2::Descriptor;
using OrbVocabulary2 = MiniBow2::TemplatedVocabulary<Descriptor>;

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



void testVocCreation(const std::vector<std::vector<Descriptor>>& features, OrbVocabulary& voc)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    auto weight = MiniBow::TF_IDF;



    voc = OrbVocabulary(k, L, weight);

    std::cout << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
    voc.create(features);
    std::cout << "Vocabulary information: " << std::endl << voc << std::endl << std::endl;


    //    exit(0);
    std::cout << "Testing loading saving..." << std::endl;

    voc.saveRaw("testvoc.minibow");
    std::cout << voc << std::endl;
    OrbVocabulary db2;
    db2.loadRaw("testvoc.minibow");
    std::cout << db2 << std::endl;
    std::cout << "... done." << std::endl << std::endl;
}



void testVocCreation(const std::vector<std::vector<Descriptor>>& features, OrbVocabulary2& voc)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;



    voc = OrbVocabulary2(k, L);

    std::cout << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
    voc.create(features);
    std::cout << "Vocabulary information: " << std::endl << voc << std::endl << std::endl;


    //    exit(0);
    std::cout << "Testing loading saving..." << std::endl;

    voc.saveRaw("testvoc.minibow");
    std::cout << voc << std::endl;
    OrbVocabulary db2;
    db2.loadRaw("testvoc.minibow");
    std::cout << db2 << std::endl;
    std::cout << "... done." << std::endl << std::endl;
}


void testVocMatching(const std::vector<std::vector<Descriptor>>& features, OrbVocabulary& voc)
{
    // lets do something with this vocabulary
    std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;

    double out = 0;

    std::vector<std::pair<MiniBow::BowVector, MiniBow::FeatureVector>> bows(features.size());

    {
        SAIGA_BLOCK_TIMER();
        for (int k = 0; k < 100; ++k)
        {
            for (int i = 0; i < features.size(); i++)
            {
                voc.transform(features[i], bows[i].first, bows[i].second, 4);
            }
        }
    }


    {
        SAIGA_BLOCK_TIMER();
        for (int k = 0; k < 100; ++k)
        {
            for (int i = 0; i < features.size(); i++)
            {
                for (int j = 0; j < features.size(); j++)
                {
                    double score = voc.score(bows[i].first, bows[j].first);
                    out += score;
                    //                std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
                }
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
    {
        SAIGA_BLOCK_TIMER();
        for (int k = 0; k < 100; ++k)
        {
            for (int i = 0; i < features.size(); i++)
            {
                voc.transform(features[i], bows[i].first, bows[i].second, 4);
            }
        }
    }


    {
        SAIGA_BLOCK_TIMER();
        for (int k = 0; k < 100; ++k)
        {
            for (int i = 0; i < features.size(); i++)
            {
                for (int j = 0; j < features.size(); j++)
                {
                    double score = voc.score(bows[i].first, bows[j].first);
                    out += score;
                    //                std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
                }
            }
        }
    }
    //        std::cout << "Score time: " << time / (features.size() * features.size()) << "ms" << std::endl;
}


TEST(Sohpus, SE3_Point)
{
    std::vector<std::vector<Descriptor>> features;
    loadFeatures(features);


    srand(23053250);
    OrbVocabulary trainedVoc;
    testVocCreation(features, trainedVoc);

    srand(23053250);
    OrbVocabulary2 trainedVoc2;
    testVocCreation(features, trainedVoc2);


    //    OrbVocabulary orbVoc("ORBvoc.minibow");
    OrbVocabulary orbVoc = trainedVoc;
    std::cout << orbVoc << std::endl;


    //    OrbVocabulary2 orbVoc2("ORBvoc.minibow");
    OrbVocabulary2 orbVoc2 = trainedVoc2;
    std::cout << orbVoc2 << std::endl;



    MiniBow::BowVector bv;
    MiniBow::FeatureVector fv;

    MiniBow2::BowVector bv2;
    MiniBow2::FeatureVector fv2;

    orbVoc.transform(features.front(), bv, fv, 4);
    orbVoc2.transform(features.front(), bv2, fv2, 4);

    EXPECT_EQ(bv.size(), bv2.size());
    EXPECT_EQ(fv.size(), fv2.size());

    {
        auto it1 = bv.begin();
        auto it2 = bv2.begin();

        while (it1 != bv.end() && it2 != bv2.end())
        {
            EXPECT_EQ(it1->first, it2->first);
            EXPECT_EQ(it1->second, it2->second);
            ++it1;
            ++it2;
        }
    }
    {
        auto it1 = fv.begin();
        auto it2 = fv2.begin();

        while (it1 != fv.end() && it2 != fv2.end())
        {
            EXPECT_EQ(it1->first, it2->first);

            // the feature reference must not be sorted
            std::sort(it1->second.begin(), it1->second.end());
            std::sort(it2->second.begin(), it2->second.end());
            EXPECT_EQ(it1->second, it2->second);
            ++it1;
            ++it2;
        }
    }


    std::cout << "Testing Matching with ORB-SLAM Voc..." << std::endl;
    testVocMatching(features, orbVoc);
    testVocMatching(features, orbVoc2);
}

}  // namespace Saiga
