/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"
#include "saiga/vision/slam/MiniBow.h"

#include <fstream>

using namespace Saiga;
using Descriptor    = MiniBow::FORB::TDescriptor;
using OrbVocabulary = MiniBow::TemplatedVocabulary<Descriptor, MiniBow::FORB, MiniBow::L1Scoring>;

const int images           = 10;
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


void testVocMatching(const std::vector<std::vector<Descriptor>>& features, OrbVocabulary& voc)
{
    // lets do something with this vocabulary
    std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;


    double out = 0;


    std::vector<std::pair<MiniBow::BowVector, MiniBow::FeatureVector>> bows(features.size());

    float time;
    {
        auto stats1 = measureObject(50, [&]() {
            for (int i = 0; i < (int)features.size(); i++)
            {
                voc.transform(features[i], bows[i].first, bows[i].second, 4);
            }
        });

        auto stats2 = measureObject(50, [&]() {

#pragma omp parallel num_threads(4)
            {
                for (int i = 0; i < (int)features.size(); i++)
                {
                    voc.transformOMP(features[i], bows[i].first, bows[i].second, 4);
                }
            }
        });

        std::cout << "Transform time: " << stats1.median / (features.size()) << "ms" << std::endl;
        std::cout << "Transform time OMP: " << stats2.median / (features.size()) << "ms" << std::endl;
    }


    {
        ScopedTimer tim(time);
        for (int i = 0; i < (int)features.size(); i++)
        {
            for (int j = 0; j < (int)features.size(); j++)
            {
                double score = voc.score(bows[i].first, bows[j].first);
                out += score;
                //                std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
            }
        }
    }
    std::cout << "Score time: " << time / (features.size() * features.size()) << "ms" << std::endl;
}

int main(int, char**)
{
    std::vector<std::vector<Descriptor>> features;
    loadFeatures(features);

    OrbVocabulary trainedVoc;
    testVocCreation(features, trainedVoc);

    std::cout << "Testing Matching with trained Voc..." << std::endl;
    testVocMatching(features, trainedVoc);

    OrbVocabulary orbVoc("ORBvoc.minibow");
    std::cout << "Testing Matching with ORB-SLAM Voc..." << std::endl;
    testVocMatching(features, orbVoc);

    return 0;
}
