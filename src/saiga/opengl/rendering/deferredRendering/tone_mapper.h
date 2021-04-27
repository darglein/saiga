/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/camera/HDR.h"
#include "saiga/core/camera/camera.h"
#include "saiga/core/util/quality.h"
#include "saiga/opengl/UnifiedMeshBuffer.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/deferredRendering/gbuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/opengl/vertex.h"
namespace Saiga
{
struct TonemapParameters
{
    vec4 vignette_coeffs = vec4::Zero();
    vec2 vignette_offset = vec2::Zero();
    float exposure = 1;

    TonemapParameters() { static_assert(sizeof(TonemapParameters) == sizeof(float) * 8); }
};

class SAIGA_OPENGL_API ToneMapper
{
   public:
    ToneMapper();
    void Map(Texture* input_hdr_color_image, Texture* output_ldr_color_image);

    void imgui();
    std::shared_ptr<Shader> shader;

    TonemapParameters params;
    TemplatedUniformBuffer<TonemapParameters> uniforms;
    bool params_dirty = true;

    std::shared_ptr<Texture1D> response_texture;
    DiscreteResponseFunction<float> camera_response;
};

}  // namespace Saiga
