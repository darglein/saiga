/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/util/Align.h"
#include "saiga/core/util/tostring.h"

namespace Saiga
{
class SAIGA_CORE_API OffModelLoader
{
   public:
    OffModelLoader() {}
    OffModelLoader(const std::string& file);
    bool loadFile(const std::string& file);

    // Output mesh
    TriangleMesh<VertexNC, uint32_t> mesh;

   private:
    enum class ParsingState
    {
        HEADER,
        COUNTS,
        VERTICES,
        FACES,
        DONE
    };
    ParsingState state = ParsingState::HEADER;
    std::string file;
};

}  // namespace Saiga
