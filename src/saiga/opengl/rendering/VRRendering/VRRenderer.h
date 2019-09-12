/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/deferredRendering/postProcessor.h"
#include "saiga/opengl/rendering/forwardRendering/forward_renderer.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/vr/VRTest.h"

namespace Saiga
{
struct SAIGA_OPENGL_API VRRenderingParameters : public RenderingParameters
{
    void fromConfigFile(const std::string& file) {}
};

class SAIGA_OPENGL_API VRRenderer : public OpenGLRenderer
{
   public:
    using InterfaceType = ForwardRenderingInterface;
    using ParameterType = VRRenderingParameters;

    VRRenderingParameters params;

    VRRenderer(OpenGLWindow& window, const VRRenderingParameters& params = VRRenderingParameters());
    virtual ~VRRenderer() {}

    virtual float getTotalRenderTime() override { return timer.getTimeMS(); }
    virtual void render(const RenderInfo& renderInfo) override;

   private:
    void renderEye(Camera* camera, vr::Hmd_Eye eye, Framebuffer& target);

    FilteredMultiFrameOpenGLTimer timer;

    // for left and right eye
    Framebuffer framebuffers[2];
    std::shared_ptr<Texture> textures[2];

    VR::OpenVRWrapper vr;

    mat4 projLeft, projRight;
    mat4 viewLeft, viewRight;

    std::shared_ptr<PostProcessingShader> framebufferToDebugWindowShader;
    IndexedVertexBuffer<VertexNT, GLushort> quadMesh;
};

}  // namespace Saiga
