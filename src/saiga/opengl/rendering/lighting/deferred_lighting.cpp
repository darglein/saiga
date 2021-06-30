/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "deferred_lighting.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/lighting/deferred_light_shader.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/CubeTexture.h"

namespace Saiga
{
DeferredLighting::DeferredLighting(GBuffer& framebuffer, GLTimerSystem* timer)
    : RendererLighting(timer), gbuffer(framebuffer)
{
    createLightMeshes();
}

DeferredLighting::~DeferredLighting() {}

void DeferredLighting::loadShaders()
{
    RendererLighting::loadShaders();

    const RendererLightingShaderNames& names = RendererLightingShaderNames();

    if (!pointLightVolumetricShader)
    {
        pointLightVolumetricShader = shaderLoader.load<PointLightShader>(names.pointLightShader, volumetricInjection);
    }

    if (!spotLightVolumetricShader)
    {
        spotLightVolumetricShader = shaderLoader.load<SpotLightShader>(names.spotLightShader, volumetricInjection);
    }

    stencilShader = shaderLoader.load<MVPShader>(names.stencilShader);
}

void DeferredLighting::init(int _width, int _height, bool _useTimers)
{
    RendererLighting::init(_width, _height, _useTimers);

    lightAccumulationBuffer.create();

    // NOTE: Use the same depth-stencil buffer as the gbuffer. I hope this works on every hardware :).
    lightAccumulationBuffer.attachTextureDepthStencil(gbuffer.getTextureDepth());

    lightAccumulationTexture = std::make_shared<Texture>();
    lightAccumulationTexture->create(_width, _height, GL_RGBA, GL_RGBA16F, GL_HALF_FLOAT);
    lightAccumulationBuffer.attachTexture(lightAccumulationTexture);

    {
        volumetricLightTexture = std::make_shared<Texture>();
        volumetricLightTexture->create(_width, _height, GL_RGBA, GL_RGBA16F, GL_HALF_FLOAT);
        lightAccumulationBuffer.attachTexture(volumetricLightTexture);
    }

    //    lightAccumulationBuffer.drawToAll();
    lightAccumulationBuffer.drawTo({0});
    lightAccumulationBuffer.check();
    lightAccumulationBuffer.unbind();


    volumetricBuffer.create();
    {
        volumetricLightTexture2 = std::make_shared<Texture>();
        volumetricLightTexture2->create(_width, _height, GL_RGBA, GL_RGBA16F, GL_HALF_FLOAT);
        volumetricBuffer.attachTexture(volumetricLightTexture2);
    }
    volumetricBuffer.drawTo({0});
    volumetricBuffer.check();
    volumetricBuffer.unbind();

    volumetricInjection.emplace_back(GL_FRAGMENT_SHADER, "#define VOLUMETRIC", 3);

    volumetricInjection.insert(volumetricInjection.end(), shadowInjection.begin(), shadowInjection.end());
}

void DeferredLighting::resize(int _width, int _height)
{
    RendererLighting::resize(_width, _height);
    lightAccumulationBuffer.resize(_width, _height);
}

void DeferredLighting::initRender()
{
    RendererLighting::initRender();

    lightAccumulationBuffer.bind();
    if (renderVolumetric)
        lightAccumulationBuffer.drawTo({0, 1});
    else
        lightAccumulationBuffer.drawTo({0});

    //    glClearColor(0,0,0,0);
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT);
    lightAccumulationBuffer.unbind();
}

void DeferredLighting::render(Camera* cam, const ViewPort& viewPort)
{
    // Does nothing
    RendererLighting::render(cam, viewPort);

    lightAccumulationBuffer.bind();

#if 0
    if (renderVolumetric)
        lightAccumulationBuffer.drawTo({0, 1});
    else
        lightAccumulationBuffer.drawTo({0});

    //    glClearColor(0,0,0,0);
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT);
#endif



    // deferred lighting uses additive blending of the lights.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    // never overwrite current depthbuffer
    glDepthMask(GL_FALSE);

    if (stencilCulling)
    {
        // all light volumes are using stencil culling
        glEnable(GL_STENCIL_TEST);
    }
    else
    {
        glDisable(GL_STENCIL_TEST);
    }

    // use depth test for all light volumes
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    //    glClearStencil(0x0);
    //    glClear(GL_STENCIL_BUFFER_BIT);
    currentStencilId = 1;

    assert_no_glerror();

    {
        auto tim = timer->Measure("Point Lights");

        if (shadowManager.point_light_shadows)
        {
            if (pointLightShadowShader->bind())
            {
                pointLightShadowShader->upload(7, shadowManager.point_light_shadows.get(), 5);
                pointLightShadowShader->unbind();
            }

            if (pointLightVolumetricShader->bind())
            {
                pointLightVolumetricShader->upload(7, shadowManager.point_light_shadows.get(), 5);
                pointLightVolumetricShader->unbind();
            }
            shadowManager.shadow_data_point_light.bind(SHADOW_DATA_BINDING_POINT);
        }

        for (auto& l : active_point_lights)
        {
            renderLightVolume(pointLightMesh, l, cam, viewPort, pointLightShader, pointLightShadowShader,
                              pointLightVolumetricShader);
        }
    }

    {
        auto tim = timer->Measure("Spot Lights");


        if (shadowManager.spot_light_shadows)
        {
            if (spotLightShadowShader->bind())
            {
                spotLightShadowShader->upload(7, shadowManager.spot_light_shadows.get(), 5);
                spotLightShadowShader->unbind();
            }

            if (spotLightVolumetricShader->bind())
            {
                spotLightVolumetricShader->upload(7, shadowManager.spot_light_shadows.get(), 5);
                spotLightVolumetricShader->unbind();
            }

            shadowManager.shadow_data_spot_light.bind(SHADOW_DATA_BINDING_POINT);
        }

        for (auto& l : active_spot_lights)
        {
            renderLightVolume(spotLightMesh, l, cam, viewPort, spotLightShader, spotLightShadowShader,
                              spotLightVolumetricShader);
        }
    }

    assert_no_glerror();

    // reset depth test to default value
    glDepthFunc(GL_LESS);



    // use default culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glDisable(GL_DEPTH_TEST);

    {
        auto tim = timer->Measure("Directional Lights");
        if (stencilCulling)
        {
            glStencilFunc(GL_NOTEQUAL, 0xFF, 0xFF);
            //    glStencilFunc(GL_EQUAL, 0x0, 0xFF);
            //    glStencilFunc(GL_ALWAYS, 0xFF, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        }


        // volumetric directional lights are not supported
        if (renderVolumetric) lightAccumulationBuffer.drawTo({0});

        if (shadowManager.cascaded_shadows)
        {
            if (directionalLightShadowShader->bind())
            {
                directionalLightShadowShader->upload(9, shadowManager.cascaded_shadows.get(), 6);
                directionalLightShadowShader->unbind();
            }
            shadowManager.shadow_data_directional_light.bind(SHADOW_DATA_BINDING_POINT);
        }

        renderDirectionalLights(cam, viewPort, false);
        renderDirectionalLights(cam, viewPort, true);
    }

    if (stencilCulling)
    {
        glDisable(GL_STENCIL_TEST);
    }
    assert_no_glerror();

    if (renderVolumetric)
    {
        auto tim = timer->Measure("Filter Volumetric");
        postprocessVolumetric();
        lightAccumulationBuffer.bind();
    }

    // reset state
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    if (drawDebug)
    {
        auto tim = timer->Measure("Debug");
        //        glDepthMask(GL_TRUE);
        renderDebug(cam);
        //        glDepthMask(GL_FALSE);
    }


    assert_no_glerror();
}

void DeferredLighting::postprocessVolumetric()
{
    // lazy load
    if (!volumetricBlurShader)
    {
        volumetricBlurShader  = shaderLoader.load<MVPTextureShader>("lighting/volumetricBlur.glsl");
        volumetricBlurShader2 = shaderLoader.load<Shader>("lighting/volumetricBlur2.glsl");
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);


    if (volumetricBlurShader->bind())
    {
        volumetricBuffer.bind();
        volumetricBlurShader->uploadModel(mat4::Identity());
        volumetricBlurShader->uploadTexture(volumetricLightTexture.get());
        directionalLightMesh.bindAndDraw();
        volumetricBuffer.unbind();
        volumetricBlurShader->unbind();
    }



#if 0
    volumetricBlurShader2->bind();
    //    volumetricLightTexture2->bind(0);
    volumetricLightTexture2->bindImageTexture(0,GL_WRITE_ONLY);
    //    glBindImageTexture( 0, volumetricLightTexture2->getId(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16 );
    //    volumetricBlurShader2->upload(5,5);
    //    std::cout << width << "x" << height << std::endl;
    volumetricBlurShader2->dispatchCompute(Saiga::iDivUp(width,16),Saiga::iDivUp(height,16),1);
    //    volumetricBlurShader2->dispatchCompute(width,height,1);
    volumetricBlurShader2->unbind();
#endif

    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    assert_no_glerror();
}



void DeferredLighting::setupStencilPass()
{
    // don't write color in stencil pass
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    // render only front faces
    glCullFace(GL_BACK);

    // default depth test
    glDepthFunc(GL_LEQUAL);

    // set stencil to 'id' if depth test fails
    // all 'failed' pixels are now marked in the stencil buffer with the id
    glStencilFunc(GL_ALWAYS, currentStencilId, 0xFF);
    //    glStencilFunc(GL_LEQUAL, currentStencilId, 0xFF);
    glStencilOp(GL_KEEP, GL_REPLACE, GL_KEEP);
}
void DeferredLighting::setupLightPass(bool isVolumetric)
{
    // write color in the light pass
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    // render only back faces
    glCullFace(GL_FRONT);

    if (lightDepthTest && !isVolumetric)
    {
        // reversed depth test: it passes if the light volume is behind an object
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_GEQUAL);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
    }

    if (stencilCulling)
    {
        // discard all pixels that are marked with 'id' from the previous pass
        glStencilFunc(GL_NOTEQUAL, currentStencilId, 0xFF);
        //    glStencilFunc(GL_NEVER, currentStencilId, 0xFF);
        //    glStencilFunc(GL_GREATER, currentStencilId, 0xFF);
        //    glStencilFunc(GL_EQUAL, 0x0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

        //-> the reverse depth test + the stencil test make now sure that the current pixel is in the light volume
        // this also works, when the camera is inside the volume, but fails when the far plane is intersecting the
        // volume


        // increase stencil id, so the next light will write a different value to the stencil buffer.
        // with this trick the expensive clear can be saved after each light
        currentStencilId++;

        if (currentStencilId == 256)
        {
            currentStencilId = 1;
            glClearStencil(0x0);
            glClear(GL_STENCIL_BUFFER_BIT);
            // we need to clear here in case we have too many lights.
        }
        SAIGA_ASSERT(currentStencilId < 256);
    }

    if (renderVolumetric && isVolumetric)
        lightAccumulationBuffer.drawTo({0, 1});
    else
        lightAccumulationBuffer.drawTo({0});
}



void DeferredLighting::renderDirectionalLights(Camera* cam, const ViewPort& vp, bool shadow)
{
    if (active_directional_lights.empty()) return;

    std::shared_ptr<DirectionalLightShader> shader = (shadow) ? directionalLightShadowShader : directionalLightShader;
    SAIGA_ASSERT(shader);
    if(shader->bind())
    {
        shader->DeferredShader::uploadFramebuffer(&gbuffer);
        shader->uploadScreenSize(vp.getVec4());
        shader->uploadSsaoTexture(ssaoTexture);



        directionalLightMesh.bind();
        for (auto& obj : active_directional_lights)
        {
            bool render =
                (shadow && obj->shouldCalculateShadowMap()) || (!shadow && obj->shouldRender() && !obj->castShadows);
            if (render)
            {
                shader->SetUniforms(obj, cam);
                directionalLightMesh.draw();
            }
        }
        directionalLightMesh.unbind();
        shader->unbind();
    }
}


void DeferredLighting::blitGbufferDepthToAccumulationBuffer()
{
    //    glEnable(GL_DEPTH_TEST);
    //    glDepthFunc(GL_ALWAYS);
    //    blitDepthShader->bind();
    //    blitDepthShader->uploadTexture(gbuffer.getTextureDepth().get());
    //    directionalLightMesh.bindAndDraw();
    //    blitDepthShader->unbind();
    //    glDepthFunc(GL_LESS);



    //    glBindFramebuffer(GL_READ_FRAMEBUFFER, gbuffer.getId());
    //    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, lightAccumulationBuffer.getId());
    //    glBlitFramebuffer(0, 0, gbuffer.getTextureDepth()->getWidth(), gbuffer.getTextureDepth()->getHeight(), 0,
    //    0, gbuffer.getTextureDepth()->getWidth(), gbuffer.getTextureDepth()->getHeight(),GL_DEPTH_BUFFER_BIT |
    //    GL_STENCIL_BUFFER_BIT, GL_NEAREST);

    //    glClearColor(0,0,0,0);
    //    glClear( GL_COLOR_BUFFER_BIT );
}

void DeferredLighting::applyVolumetricLightBuffer()
{
    if (!renderVolumetric) return;

    if (!textureShader) textureShader = shaderLoader.load<MVPTextureShader>("lighting/light_test.glsl");

    auto tim = timer->Measure("Blend Volumetric");
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    lightAccumulationBuffer.bind();

    lightAccumulationBuffer.drawTo({0});


    if(textureShader->bind())
    {
        textureShader->uploadModel(mat4::Identity());
        textureShader->uploadTexture(volumetricLightTexture2.get());
        directionalLightMesh.bindAndDraw();
        textureShader->unbind();
    }

    assert_no_glerror();
}


void DeferredLighting::setStencilShader(std::shared_ptr<MVPShader> stencilShader)
{
    this->stencilShader = stencilShader;
}


// std::shared_ptr<DirectionalLight> DeferredLighting::createDirectionalLight()
//{
//    std::shared_ptr<DirectionalLight> l = std::make_shared<DirectionalLight>();
//    directionalLights.insert(l);
//    return l;
//}

// std::shared_ptr<PointLight> DeferredLighting::createPointLight()
//{
//    std::shared_ptr<PointLight> l = std::make_shared<PointLight>();
//    pointLights.insert(l);
//    return l;
//}

// std::shared_ptr<SpotLight> DeferredLighting::createSpotLight()
//{
//    std::shared_ptr<SpotLight> l = std::make_shared<SpotLight>();
//    spotLights.insert(l);
//    return l;
//}


void DeferredLighting::renderImGui()
{
    if (!showLightingImgui) return;

    if (!editor_gui.enabled)
    {
        int w = 340;
        int h = 240;
        ImGui::SetNextWindowPos(ImVec2(680, height - h), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_Once);
    }

    if (ImGui::Begin("Lighting", &showLightingImgui))
    {
        ImGui::Text("Deferred Light Volumes");
        ImGui::Checkbox("stencilCulling", &stencilCulling);
        ImGui::Separator();
    }
    ImGui::End();


    RendererLighting::renderImGui();
}

}  // namespace Saiga
