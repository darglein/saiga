/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/deferredRendering/postProcessor.h"
#include "saiga/opengl/rendering/forwardRendering/forward_renderer.h"
#include "saiga/opengl/rendering/deferredRendering/deferred_renderer.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/vr/OpenvrWrapper.h"

namespace Saiga
{
struct SAIGA_OPENGL_API VRRenderingParameters : public DeferredRenderingParameters
{
    void fromConfigFile(const std::string& file) {}
};

class SAIGA_OPENGL_API VRRenderer : public DeferredRenderer
{
   public:
    using InterfaceType = RenderingInterface;
    using ParameterType = VRRenderingParameters;

    VRRenderingParameters params;

    VRRenderer(OpenGLWindow& window, const VRRenderingParameters& params = VRRenderingParameters());
    virtual ~VRRenderer() {}

    virtual void render(const RenderInfo& renderInfo) override;

    OpenVRWrapper& VR() { return *vr; }

   private:
    // for left and right eye
    Framebuffer framebuffers[2];
    std::shared_ptr<Texture> textures[2];

    std::shared_ptr<OpenVRWrapper> vr;

    mat4 projLeft, projRight;
    mat4 viewLeft, viewRight;

    std::shared_ptr<PostProcessingShader> framebufferToDebugWindowShader;
    UnifiedMeshBuffer quadMesh;
};

}  // namespace Saiga
