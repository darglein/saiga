
// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: darius.rueckert@fau.de (Darius Rueckert)

#ifndef CERES_PUBLIC_AutoDiffCodeGen_H_
#define CERES_PUBLIC_AutoDiffCodeGen_H_

#include "ExpressionJet.h"
#include "ceres/internal/autodiff.h"
#include "ceres/types.h"

#include "ceres/autodiff_cost_function.h"
#include "ceres/sized_cost_function.h"

namespace ceres
{
template <typename CostFunctor, int kNumResiduals, int... Ns>
struct AutoDiffCodeGen : public SizedCostFunction<kNumResiduals, Ns...>
{
    explicit AutoDiffCodeGen(CostFunctor* functor) : functor_(functor) {}

   public:
    bool Generate()
    {
        using T             = double;
        using ParameterDims = typename SizedCostFunction<kNumResiduals, Ns...>::ParameterDims;
        typedef ExpressionJet<T, ParameterDims::kNumParameters> JetT;

        auto num_outputs = SizedCostFunction<kNumResiduals, Ns...>::num_residuals();
        internal::FixedArray<JetT, (256 * 7) / sizeof(JetT)> x(ParameterDims::kNumParameters + num_outputs);

        using Parameters = typename ParameterDims::Parameters;

        // These are the positions of the respective jets in the fixed array x.
        std::array<JetT*, ParameterDims::kNumParameterBlocks> unpacked_parameters =
            ParameterDims::GetUnpackedParameters(x.data());
        JetT* output = x.data() + ParameterDims::kNumParameters;

        int totalParamId = 0;
        for (int i = 0; i < ParameterDims::kNumParameterBlocks; ++i)
        {
            for (int j = 0; j < ParameterDims::GetDim(i); ++j)
            {
                JetT& J = x[totalParamId];
                J.a =
                    JetT::factory.ParameterExpr(i, "parameters[" + std::to_string(i) + "][" + std::to_string(j) + "]");
                J.v = JetT::factory.template ConstantExprArray<ParameterDims::kNumParameters>(totalParamId);
                totalParamId++;
            }
        }

        if (!internal::VariadicEvaluate<ParameterDims>(*functor_, unpacked_parameters.data(), output))
        {
            return false;
        }

        // the (non-dervied) function
        CodeFunction f;
        f.removedUnusedParameters = false;
        f.factory                 = JetT::factory;

        for (int i = 0; i < num_outputs; ++i)
        {
            auto& J  = output[i];
            auto res = f.factory.OutputAssignExpr(J.a, "residuals[" + std::to_string(i) + "]");
            f.targets.push_back(res);
        }
#if 1
        totalParamId = 0;
        for (int i = 0; i < ParameterDims::kNumParameterBlocks; ++i)
        {
            for (int j = 0; j < ParameterDims::GetDim(i); ++j)
            {
                for (int r = 0; r < num_outputs; ++r)
                {
                    auto& J = output[r];
                    // all partial derivatives
                    //            for (int j = 0; j < ParameterDims::kNumParameters; ++j)
                    //            {
                    auto res = f.factory.OutputAssignExpr(J.v[totalParamId],
                                                          "jacobians[" + std::to_string(i) + "][" +
                                                              std::to_string(r * ParameterDims::GetDim(i) + j) + "]");
                    f.targets.push_back(res);
                }
                totalParamId++;
            }
        }
#endif

        f.generate();
        f.print();


        return true;
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        return false;
    }


    std::unique_ptr<CostFunctor> functor_;
};

}  // namespace ceres
#endif
