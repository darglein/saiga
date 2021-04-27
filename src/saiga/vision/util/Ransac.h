/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/VisionTypes.h"


namespace Saiga
{
// This seed is used for all ransac classes.
// You can change this in your application for example to:
// ransacSeed = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
SAIGA_VISION_API extern uint64_t ransacRandomSeed;



struct RansacParameters
{
    int maxIterations = -1;

    // compared to the value which is returned from computeResidual.
    // usually you want to return the squared norm
    double residualThreshold;

    // expected maximum number of N
    // internal data structures will be reserved to this size
    int reserveN = 0;

    // Number of omp threads in that group
    // Note:
    int threads = 1;
};


template <typename Derived, typename Model, int ModelSize>
class RansacBase
{
   public:
    void init(const RansacParameters& _params)
    {
        params = _params;
        SAIGA_ASSERT(params.maxIterations > 0);
        SAIGA_ASSERT(OMP::getNumThreads() == 1);

        numInliers.resize(params.maxIterations);
        models.resize(params.maxIterations);

        residuals.resize(params.maxIterations);
        for (auto&& r : residuals) r.reserve(params.reserveN);

        inliers.resize(params.maxIterations);
        for (auto&& r : inliers) r.reserve(params.reserveN);

        SAIGA_ASSERT(params.threads >= 1);
        generators.resize(params.threads);
        threadLocalBestModel.resize(params.threads);
        for (int i = 0; i < params.threads; ++i)
        {
            generators[i].seed(ransacRandomSeed + 6643838879UL * i);
        }
    }

    const RansacParameters& Params() const { return params; }

   protected:
    // indices of subset
    using Subset = std::array<int, ModelSize>;

    RansacBase() {}
    RansacBase(const RansacParameters& _params) { init(_params); }



    int compute(int _N)
    {
        SAIGA_ASSERT(params.maxIterations > 0);
        SAIGA_ASSERT(OMP::getNumThreads() == params.threads);

        int tid = OMP::getThreadNum();
        // compute random sample subsets
        std::uniform_int_distribution<int> dis(0, _N - 1);
        auto& gen = generators[tid];


        auto& bestModel = threadLocalBestModel[tid]();
        bestModel       = {0, 0};


#pragma omp for
        for (int it = 0; it < params.maxIterations; ++it)
        {
            auto& model     = models[it];
            auto& inlier    = inliers[it];
            auto& residual  = residuals[it];
            auto& numInlier = numInliers[it];

            numInlier = 0;
            residual.resize(_N);
            inlier.resize(_N);

            Subset set;
            for (auto j : Range(0, ModelSize))
            {
                auto idx = dis(gen);
                set[j]   = idx;
            }

            if (!derived().computeModel(set, model)) continue;

            for (int j = 0; j < _N; ++j)
            {
                residual[j] = derived().computeResidual(model, j);

                bool inl  = residual[j] < params.residualThreshold;
                inlier[j] = inl;
                numInlier += inl;
            }

            if (numInlier > bestModel.first)
            {
                bestModel.first  = numInlier;
                bestModel.second = it;
            }
        }

#pragma omp single
        {
            N             = _N;
            bestIdx       = 0;
            int bestCount = 0;
            for (int th = 0; th < params.threads; ++th)
            {
                auto&& thbestModel = threadLocalBestModel[th]();
                auto inl           = thbestModel.first;
                auto it            = thbestModel.second;
                //                std::cout << "th best " << th << " " << it << " " << inl << std::endl;
                if (inl > bestCount)
                {
                    bestCount = inl;
                    bestIdx   = it;
                }
            }
        }
        return bestIdx;
    }


    // total number of sample points
    int N;
    RansacParameters params;
    AlignedVector<std::vector<double>> residuals;
    AlignedVector<std::vector<char>> inliers;
    AlignedVector<int> numInliers;
    AlignedVector<Model> models;

    // make sure we don't run into false sharing
    AlignedVector<AlignedStruct<std::pair<int, int>, SAIGA_CACHE_LINE_SIZE>> threadLocalBestModel;

    // each thread has one generator
    std::vector<std::mt19937> generators;

    int bestIdx;

   private:
    Derived& derived() { return *static_cast<Derived*>(this); }
};

inline int RansacIterationsFromProbability(int input_N, double probability, int minInliers, int maxIterations)
{
    // Adjust Parameters according to number of correspondences
    float epsilon = (float)minInliers / input_N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if (minInliers >= input_N)
        nIterations = 1;
    else
        nIterations = ceil(log(1 - probability) / log(1 - pow(epsilon, 3)));

    nIterations = std::max(1, std::min(nIterations, maxIterations));
    return nIterations;
}

// double inlierProb  = 0.7;
// double successProb = 0.999;

// double k = log(1 - successProb) / log(1 - pow(inlierProb, 5));
// std::cout << k << std::endl;



}  // namespace Saiga
