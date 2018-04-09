/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/color.h"
#include <saiga/assets/coloredAsset.h>
#include <saiga/assets/animatedAsset.h>
#include "saiga/animation/boneShader.h"

namespace Saiga {

class SAIGA_GLOBAL AssetLoader2{
public:
    std::shared_ptr<MVPShader> basicAssetShader;
    std::shared_ptr<MVPShader> basicAssetForwardShader;
    std::shared_ptr<MVPShader> basicAssetDepthshader;
    std::shared_ptr<MVPShader> basicAssetWireframeShader;

    std::shared_ptr<MVPShader> texturedAssetShader;
    std::shared_ptr<MVPShader> texturedAssetForwardShader;
    std::shared_ptr<MVPShader> texturedAssetDepthShader;
    std::shared_ptr<MVPShader> texturedAssetWireframeShader;

    std::shared_ptr<BoneShader> animatedAssetShader;
    std::shared_ptr<BoneShader> animatedAssetForwardShader;
    std::shared_ptr<BoneShader> animatedAssetDepthshader;
    std::shared_ptr<BoneShader> animatedAssetWireframeShader;

    AssetLoader2();
    virtual ~AssetLoader2();

    void loadDefaultShaders();

    /**
     * Creates a plane with a checker board texture.
     * The plane lays in the x-z plane with a normal pointing to positve y.
     * size.x and size.y are the dimensions of the plane.
     * quadSize is the size of one individual quad of the checkerboard.
     */

    std::shared_ptr<TexturedAsset> loadDebugPlaneAsset(vec2 size, float quadSize=1.0f, Color color1=Colors::lightgray, Color color2=Colors::gray);

    std::shared_ptr<TexturedAsset> loadDebugTexturedPlane(std::shared_ptr<Texture> texture, vec2 size);

    std::shared_ptr<ColoredAsset> loadDebugArrow(float radius, float length, vec4 color=vec4(1,0,0,1));

    std::shared_ptr<ColoredAsset> assetFromMesh(std::shared_ptr<TriangleMesh<VertexNT,GLuint>> mesh, const vec4& color=vec4(1,1,1,1));
    std::shared_ptr<ColoredAsset> assetFromMesh(TriangleMesh<VertexNT,GLuint>& mesh, const vec4& color=vec4(1,1,1,1));

    std::shared_ptr<ColoredAsset> nonTriangleMesh(std::vector<vec3> vertices, std::vector<GLuint> indices, GLenum mode = GL_TRIANGLES, const vec4& color=vec4(1,1,1,1));

    std::shared_ptr<ColoredAsset> frustumMesh(const mat4& proj, const vec4& color=vec4(1,1,1,1));
};

}
