/**
 * Copyright (c) 2021 Darius Rückert
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
#include "saiga/opengl/shaderStorageBuffer.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/opengl/vertex.h"
namespace Saiga
{

enum class TonemapOperator : int
{
    GAMMA = 0,
    TEXTURE,
    REINHARD,
    UE3,
    UC2,
    DRAGO,
};


struct TonemapParameters
{
    vec3 white_point     = vec3::Ones();
    float exposure_value = 0;
    vec4 vignette_coeffs = vec4::Zero();
    vec2 vignette_offset = vec2::Zero();
    uint32_t flags       = 0;
    float padding;

    TonemapParameters() { static_assert(sizeof(TonemapParameters) == sizeof(float) * 12); }
};

// Recomputed dynamically
struct TonemapTempParameters
{
    // (r,g,b,l)
    vec4 average_color_luminace = vec4::Zero();
};

class SAIGA_OPENGL_API ToneMapper
{
   public:
    ToneMapper(GLenum input_type = GL_RGBA16F);

    void MapLinear(Texture* input_hdr_color_image);
    void Map(Texture* input_hdr_color_image, Texture* output_ldr_color_image);

    void imgui();
    std::shared_ptr<Shader> shader, shader_linear, average_brightness_shader;


    TonemapParameters params;
    TemplatedUniformBuffer<TonemapParameters> uniforms;

    TonemapTempParameters tmp_params;
    TemplatedShaderStorageBuffer<TonemapTempParameters> tmp_buffer;

    bool params_dirty = true;

    std::shared_ptr<Texture1D> response_texture;
    DiscreteResponseFunction<float> camera_response;

    bool auto_exposure      = false;
    bool auto_white_balance = false;

    // This two values are only valid if the flags (above) are set, Map(..) has been called and
    // the download auto update is true
    bool download_tmp_values  = false;
    float computed_exposure   = 0;
    float gamma               = 1 / 2.2;
    int tm_operator = 0;
    vec3 computed_white_point = vec3::Zero();

   private:
    void ComputeOptimalExposureValue(Texture* input_hdr_color_image);

    float color_temperature = 5500;
    GLenum input_type;
};

}  // namespace Saiga
