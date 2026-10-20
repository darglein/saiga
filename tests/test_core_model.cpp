/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/model/UnifiedModel.h"

#include "gtest/gtest.h"

#include <sstream>

using namespace Saiga;

// Regression test: operator<< of UnifiedMaterial and UnifiedModel wrote to std::cout instead of the given stream.
TEST(UnifiedModel, StreamOperatorsWriteToStream)
{
    UnifiedMaterial mat("my_material");
    std::stringstream strm;
    strm << mat;
    EXPECT_NE(strm.str().find("my_material"), std::string::npos);

    UnifiedModel model;
    model.name = "my_model";
    model.materials.push_back(mat);
    std::stringstream strm2;
    strm2 << model;
    EXPECT_NE(strm2.str().find("my_model"), std::string::npos);
    EXPECT_NE(strm2.str().find("Materials"), std::string::npos);
    EXPECT_NE(strm2.str().find("my_material"), std::string::npos);
}
