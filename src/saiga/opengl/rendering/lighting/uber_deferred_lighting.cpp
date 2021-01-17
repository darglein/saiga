/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "uber_deferred_lighting.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/CubeTexture.h"

namespace Saiga
{
using namespace uber;

UberDeferredLighting::UberDeferredLighting(GBuffer& framebuffer) : gbuffer(framebuffer)
{
    createLightMeshes();
    shadowCameraBuffer.createGLBuffer(nullptr, sizeof(CameraDataGLSL), GL_DYNAMIC_DRAW);

    int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

    maximumNumberOfDirectionalLights =
        std::clamp(maximumNumberOfDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLightData));
    maximumNumberOfPointLights = std::clamp(maximumNumberOfPointLights, 0, maxSize / (int)sizeof(PointLightData));
    maximumNumberOfSpotLights  = std::clamp(maximumNumberOfSpotLights, 0, maxSize / (int)sizeof(SpotLightData));
    maximumNumberOfBoxLights   = std::clamp(maximumNumberOfBoxLights, 0, maxSize / (int)sizeof(BoxLightData));

    lightDataBufferDirectional.createGLBuffer(nullptr, sizeof(DirectionalLightData) * maximumNumberOfDirectionalLights,
                                              GL_DYNAMIC_DRAW);
    lightDataBufferPoint.createGLBuffer(nullptr, sizeof(PointLightData) * maximumNumberOfPointLights, GL_DYNAMIC_DRAW);
    lightDataBufferSpot.createGLBuffer(nullptr, sizeof(SpotLightData) * maximumNumberOfSpotLights, GL_DYNAMIC_DRAW);
    lightDataBufferBox.createGLBuffer(nullptr, sizeof(BoxLightData) * maximumNumberOfBoxLights, GL_DYNAMIC_DRAW);
    lightInfoBuffer.createGLBuffer(nullptr, sizeof(LightInfo), GL_DYNAMIC_DRAW);


    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    quadMesh.fromMesh(*qb);

    ClustererParameters params;
    lightClusterer = std::make_shared<Clusterer>(params);
}

void UberDeferredLighting::init(int _width, int _height, bool _useTimers)
{
    RendererLighting::init(_width, _height, _useTimers);
    lightClusterer->init(_width, _height, _useTimers);
}

void UberDeferredLighting::resize(int _width, int _height)
{
    RendererLighting::resize(_width, _height);
    lightClusterer->resize(_width, _height);
}

UberDeferredLighting::~UberDeferredLighting() {}

void UberDeferredLighting::loadShaders()
{
    RendererLighting::loadShaders();

    const RendererLightingShaderNames& names = RendererLightingShaderNames();

    ShaderPart::ShaderCodeInjections sci;

    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfDirectionalLights), 1);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfPointLights), 2);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfSpotLights), 3);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_BL_COUNT" + std::to_string(maximumNumberOfBoxLights), 4);
    lightingShader = shaderLoader.load<UberDeferredLightingShader>(names.lightingUberShader, sci);
}

void UberDeferredLighting::initRender()
{
    // TODO Paul: We should refactor this for all single light pass renderers.
    startTimer(0);
    RendererLighting::initRender();
    lightDataBufferPoint.bind(POINT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferSpot.bind(SPOT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferBox.bind(BOX_LIGHT_DATA_BINDING_POINT);
    lightDataBufferDirectional.bind(DIRECTIONAL_LIGHT_DATA_BINDING_POINT);
    lightInfoBuffer.bind(LIGHT_INFO_BINDING_POINT);
    LightInfo li;
    LightData ld;
    li.pointLightCount       = 0;
    li.spotLightCount        = 0;
    li.boxLightCount         = 0;
    li.directionalLightCount = 0;

    std::vector<PointLightClusterData> plc;
    std::vector<SpotLightClusterData> slc;
    std::vector<BoxLightClusterData> blc;
    if (lightClusterer)
    {
        plc = lightClusterer->pointLightClusterData();
        slc = lightClusterer->spotLightClusterData();
        blc = lightClusterer->boxLightClusterData();
    }

    // Point Lights
    PointLightData glPointLight;
    PointLightClusterData clPointLight;
    for (auto pl : pointLights)
    {
        if (li.pointLightCount >= maximumNumberOfPointLights) break;  // just ignore too many lights...
        if (!pl->shouldRender()) continue;
        glPointLight.position      = make_vec4(pl->getPosition(), 0.0f);
        glPointLight.colorDiffuse  = make_vec4(pl->getColorDiffuse(), pl->getIntensity());
        glPointLight.colorSpecular = make_vec4(pl->getColorSpecular(), 1.0f);  // specular Intensity?
        glPointLight.attenuation   = make_vec4(pl->getAttenuation(), pl->getRadius());
        ld.pointLights.push_back(glPointLight);

        if (lightClusterer)
        {
            clPointLight.world_center = pl->getPosition();
            clPointLight.radius       = pl->getRadius();
            plc.push_back(clPointLight);
        }

        li.pointLightCount++;
    }

    // Spot Lights
    SpotLightData glSpotLight;
    SpotLightClusterData clSpotLight;
    for (auto sl : spotLights)
    {
        if (li.spotLightCount >= maximumNumberOfSpotLights) break;  // just ignore too many lights...
        if (!sl->shouldRender()) continue;
        float cosa                = cos(radians(sl->getAngle() * 0.95f));  // make border smoother
        glSpotLight.position      = make_vec4(sl->getPosition(), cosa);
        glSpotLight.colorDiffuse  = make_vec4(sl->getColorDiffuse(), sl->getIntensity());
        glSpotLight.colorSpecular = make_vec4(sl->getColorSpecular(), 1.0f);  // specular Intensity?
        glSpotLight.attenuation   = make_vec4(sl->getAttenuation(), sl->getRadius());
        glSpotLight.direction     = make_vec4(0);
        glSpotLight.direction += sl->getModelMatrix().col(1);
        ld.spotLights.push_back(glSpotLight);

        if (lightClusterer)
        {
            clSpotLight.world_center = sl->getPosition() + vec3(glSpotLight.direction.x(), glSpotLight.direction.y(),
                                                                glSpotLight.direction.z()) *
                                                               sl->getRadius() * 0.5;
            clSpotLight.radius = sl->getRadius() * 0.5;  // TODO Paul: Is that correct?
            slc.push_back(clSpotLight);
        }

        li.spotLightCount++;
    }

    // Box Lights
    BoxLightData glBoxLight;
    BoxLightClusterData clBoxLight;
    const mat4 biasMatrix = make_mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0);
    for (auto bl : boxLights)
    {
        if (li.boxLightCount >= maximumNumberOfBoxLights) break;  // just ignore too many lights...
        if (!bl->shouldRender()) continue;
        glBoxLight.colorDiffuse  = make_vec4(bl->getColorDiffuse(), bl->getIntensity());
        glBoxLight.colorSpecular = make_vec4(bl->getColorSpecular(), 1.0f);  // specular Intensity?
        glBoxLight.direction     = make_vec4(0);
        glBoxLight.direction += bl->getModelMatrix().col(2);
        bl->calculateCamera();
        glBoxLight.lightMatrix = biasMatrix * bl->shadowCamera.proj * bl->shadowCamera.view;
        ld.boxLights.push_back(glBoxLight);

        if (lightClusterer)
        {
            clBoxLight.world_center = vec3(bl->position.x(), bl->position.y(), bl->position.z());
            clBoxLight.radius =
                std::max(bl->scale.x(), std::max(bl->scale.y(), bl->scale.z()));  // TODO Paul: Is that correct?
            blc.push_back(clBoxLight);
        }

        li.boxLightCount++;
    }

    // Directional Lights
    DirectionalLightData glDirectionalLight;
    for (auto dl : directionalLights)
    {
        if (li.directionalLightCount >= maximumNumberOfDirectionalLights) break;  // just ignore too many lights...
        if (!dl->shouldRender()) continue;
        glDirectionalLight.position      = make_vec4(dl->getPosition(), 0.0f);
        glDirectionalLight.colorDiffuse  = make_vec4(dl->getColorDiffuse(), dl->getIntensity());
        glDirectionalLight.colorSpecular = make_vec4(dl->getColorSpecular(), 1.0f);  // specular Intensity?
        glDirectionalLight.direction     = make_vec4(dl->getDirection(), 0.0f);
        ld.directionalLights.push_back(glDirectionalLight);
        li.directionalLightCount++;
    }

    lightDataBufferPoint.updateBuffer(ld.pointLights.data(), sizeof(PointLightData) * li.pointLightCount, 0);
    lightDataBufferSpot.updateBuffer(ld.spotLights.data(), sizeof(SpotLightData) * li.spotLightCount, 0);
    lightDataBufferBox.updateBuffer(ld.boxLights.data(), sizeof(BoxLightData) * li.boxLightCount, 0);
    lightDataBufferDirectional.updateBuffer(ld.directionalLights.data(),
                                            sizeof(DirectionalLightData) * li.directionalLightCount, 0);

    lightInfoBuffer.updateBuffer(&li, sizeof(LightInfo), 0);
    visibleLights = li.pointLightCount + li.spotLightCount + li.boxLightCount + li.directionalLightCount;
    stopTimer(0);
}

void UberDeferredLighting::render(Camera* cam, const ViewPort& viewPort)
{
    // Does nothing
    RendererLighting::render(cam, viewPort);
    if (lightClusterer)
    {
        lightClusterer->clusterLights(cam, viewPort);
        bool is_dirty;
        const std::unordered_map<LightID, ClusterID> clusteredLightData =
            lightClusterer->getLightToClusterMap(is_dirty);
        if (is_dirty)
        {
            // This does only happen if clusterLights was not called beforehand.
        }
        // At this point we can use clustering information in the lighting uber shader.
    }


    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Lighting Uber Shader
    lightingShader->bind();
    lightingShader->uploadFramebuffer(&gbuffer);
    lightingShader->uploadScreenSize(viewPort.getVec4());
    lightingShader->uploadInvProj(inverse(cam->proj));
    quadMesh.bindAndDraw();
    lightingShader->unbind();
    assert_no_glerror();

    // reset state
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    if (drawDebug)
    {
        // glDepthMask(GL_TRUE);
        renderDebug(cam);
        // glDepthMask(GL_FALSE);
    }
    assert_no_glerror();
}

void UberDeferredLighting::setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights,
                                          int maxBoxLights)
{
    maxDirectionalLights = std::max(0, maxDirectionalLights);
    maxPointLights       = std::max(0, maxPointLights);
    maxSpotLights        = std::max(0, maxSpotLights);
    maxBoxLights         = std::max(0, maxBoxLights);

    int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

    maximumNumberOfDirectionalLights =
        std::clamp(maximumNumberOfDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLightData));
    maximumNumberOfPointLights = std::clamp(maximumNumberOfPointLights, 0, maxSize / (int)sizeof(PointLightData));
    maximumNumberOfSpotLights  = std::clamp(maximumNumberOfSpotLights, 0, maxSize / (int)sizeof(SpotLightData));
    maximumNumberOfBoxLights   = std::clamp(maximumNumberOfBoxLights, 0, maxSize / (int)sizeof(BoxLightData));


    if (maximumNumberOfDirectionalLights != maxDirectionalLights)
    {
        lightDataBufferDirectional.createGLBuffer(nullptr, sizeof(DirectionalLightData) * maxDirectionalLights,
                                                  GL_DYNAMIC_DRAW);
    }
    if (maximumNumberOfPointLights != maxPointLights)
    {
        lightDataBufferPoint.createGLBuffer(nullptr, sizeof(PointLightData) * maxPointLights, GL_DYNAMIC_DRAW);
    }
    if (maximumNumberOfSpotLights != maxSpotLights)
    {
        lightDataBufferSpot.createGLBuffer(nullptr, sizeof(SpotLightData) * maxSpotLights, GL_DYNAMIC_DRAW);
    }
    if (maximumNumberOfBoxLights != maxBoxLights)
    {
        lightDataBufferBox.createGLBuffer(nullptr, sizeof(BoxLightData) * maxBoxLights, GL_DYNAMIC_DRAW);
    }

    maximumNumberOfDirectionalLights = maxDirectionalLights;
    maximumNumberOfPointLights       = maxPointLights;
    maximumNumberOfSpotLights        = maxSpotLights;
    maximumNumberOfBoxLights         = maxBoxLights;

    const RendererLightingShaderNames& names = RendererLightingShaderNames();

    ShaderPart::ShaderCodeInjections sci;
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfDirectionalLights), 1);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfPointLights), 2);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfSpotLights), 3);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_BL_COUNT" + std::to_string(maximumNumberOfBoxLights), 4);
    lightingShader = shaderLoader.load<UberDeferredLightingShader>(names.lightingUberShader, sci);
}

void UberDeferredLighting::renderImGui(bool* p_open)
{
    RendererLighting::renderImGui(p_open);
}

}  // namespace Saiga
