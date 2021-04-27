/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#ifndef SAIGA_USE_YAML_CPP
#    error Saiga was build without libyamlcpp!
#endif

#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "yaml-cpp/yaml.h"
// This file includes a few helper function to load yaml files.
namespace Saiga
{
template <typename MatrixType>
MatrixType readYamlMatrix(const YAML::Node& node)
{
    MatrixType matrix;
    std::vector<double> data;
    for (auto n : node)
    {
        data.push_back(n.as<double>());
    }
    SAIGA_ASSERT(data.size() == (MatrixType::RowsAtCompileTime * MatrixType::ColsAtCompileTime));
    for (int i = 0; i < MatrixType::RowsAtCompileTime; ++i)
    {
        for (int j = 0; j < MatrixType::ColsAtCompileTime; ++j)
        {
            matrix(i, j) = data[i * MatrixType::ColsAtCompileTime + j];
        }
    }
    return matrix;
}

}  // namespace Saiga
