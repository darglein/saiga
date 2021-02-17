/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/color.h"
#include "saiga/opengl/animation/boneShader.h"

#include "animatedAsset.h"
#include "coloredAsset.h"

namespace Saiga
{
class SAIGA_OPENGL_API AssetLoader
{
   public:
    AssetLoader();
    virtual ~AssetLoader();


    std::shared_ptr<ColoredAsset> loadDebugArrow(float radius, float length, vec4 color = vec4(1, 0, 0, 1));

    std::shared_ptr<ColoredAsset> assetFromMesh(TriangleMesh<VertexNC, GLuint>& mesh);
    std::shared_ptr<ColoredAsset> assetFromMesh(TriangleMesh<VertexNT, GLuint>& mesh,
                                                const vec4& color = vec4(1, 1, 1, 1));

    std::shared_ptr<ColoredAsset> nonTriangleMesh(std::vector<vec3> vertices, std::vector<GLuint> indices,
                                                  GLenum mode = GL_TRIANGLES, const vec4& color = vec4(1, 1, 1, 1));

    std::shared_ptr<ColoredAsset> frustumMesh(const mat4& proj, const vec4& color = vec4(1, 1, 1, 1));
};

}  // namespace Saiga
