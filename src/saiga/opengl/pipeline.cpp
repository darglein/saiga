/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/pipeline.h"

#include "saiga/core/util/assert.h"
#include "saiga/opengl/opengl_helper.h"
#include "saiga/opengl/shader/shader.h"

#include <sstream>

namespace Saiga
{
namespace
{
struct Cache
{
    bool enabled    = false;
    bool validation = false;
    // `state` and `stencilReference` are what applyPipelineState set last. With `valid`, the context still has them:
    // the cache may skip the values that don't change.
    bool valid                = false;
    PipelineState state       = {};
    uint32_t stencilReference = 0;
    // Whether the context has GL_PRIMITIVE_RESTART_FIXED_INDEX, -1 before the first check.
    int fixedIndexRestart = -1;
    ClipSpace clipSpace   = ClipSpace::OpenGL;
};

Cache& cache()
{
    thread_local Cache c;
    return c;
}

// GL_PRIMITIVE_RESTART_FIXED_INDEX is core since OpenGL 4.3, but saiga also runs on 3.3. Checked once per thread: all
// contexts of a thread are assumed to have the same version.
bool fixedIndexRestartSupported()
{
    int& supported = cache().fixedIndexRestart;
    if (supported < 0)
    {
        const int version = getVersionMajor() * 10 + getVersionMinor();
        supported         = version >= 43 || hasExtension("GL_ARB_ES3_compatibility") ? 1 : 0;
    }
    return supported == 1;
}

GLenum toGL(CompareOp op)
{
    switch (op)
    {
        case CompareOp::Never:
            return GL_NEVER;
        case CompareOp::Less:
            return GL_LESS;
        case CompareOp::Equal:
            return GL_EQUAL;
        case CompareOp::LessOrEqual:
            return GL_LEQUAL;
        case CompareOp::Greater:
            return GL_GREATER;
        case CompareOp::NotEqual:
            return GL_NOTEQUAL;
        case CompareOp::GreaterOrEqual:
            return GL_GEQUAL;
        case CompareOp::Always:
            return GL_ALWAYS;
    }
    return GL_ALWAYS;
}

GLenum toGL(StencilOp op)
{
    switch (op)
    {
        case StencilOp::Keep:
            return GL_KEEP;
        case StencilOp::Zero:
            return GL_ZERO;
        case StencilOp::Replace:
            return GL_REPLACE;
        case StencilOp::IncrementAndClamp:
            return GL_INCR;
        case StencilOp::DecrementAndClamp:
            return GL_DECR;
        case StencilOp::Invert:
            return GL_INVERT;
        case StencilOp::IncrementAndWrap:
            return GL_INCR_WRAP;
        case StencilOp::DecrementAndWrap:
            return GL_DECR_WRAP;
    }
    return GL_KEEP;
}

GLenum toGL(BlendFactor f)
{
    switch (f)
    {
        case BlendFactor::Zero:
            return GL_ZERO;
        case BlendFactor::One:
            return GL_ONE;
        case BlendFactor::SrcColor:
            return GL_SRC_COLOR;
        case BlendFactor::OneMinusSrcColor:
            return GL_ONE_MINUS_SRC_COLOR;
        case BlendFactor::DstColor:
            return GL_DST_COLOR;
        case BlendFactor::OneMinusDstColor:
            return GL_ONE_MINUS_DST_COLOR;
        case BlendFactor::SrcAlpha:
            return GL_SRC_ALPHA;
        case BlendFactor::OneMinusSrcAlpha:
            return GL_ONE_MINUS_SRC_ALPHA;
        case BlendFactor::DstAlpha:
            return GL_DST_ALPHA;
        case BlendFactor::OneMinusDstAlpha:
            return GL_ONE_MINUS_DST_ALPHA;
    }
    return GL_ONE;
}

GLenum toGL(BlendOp op)
{
    switch (op)
    {
        case BlendOp::Add:
            return GL_FUNC_ADD;
        case BlendOp::Subtract:
            return GL_FUNC_SUBTRACT;
        case BlendOp::ReverseSubtract:
            return GL_FUNC_REVERSE_SUBTRACT;
        case BlendOp::Min:
            return GL_MIN;
        case BlendOp::Max:
            return GL_MAX;
    }
    return GL_FUNC_ADD;
}

GLenum toGL(PolygonMode mode)
{
    switch (mode)
    {
        case PolygonMode::Fill:
            return GL_FILL;
        case PolygonMode::Line:
            return GL_LINE;
        case PolygonMode::Point:
            return GL_POINT;
    }
    return GL_FILL;
}

GLenum toGL(CullMode mode)
{
    switch (mode)
    {
        case CullMode::Front:
            return GL_FRONT;
        case CullMode::FrontAndBack:
            return GL_FRONT_AND_BACK;
        case CullMode::None:
        case CullMode::Back:
            return GL_BACK;
    }
    return GL_BACK;
}

template <typename Enum, int N>
Enum fromGL(GLenum value, const Enum (&values)[N])
{
    for (Enum e : values)
    {
        if (toGL(e) == value) return e;
    }
    return values[0];
}

void setCapability(GLenum capability, bool enable)
{
    if (enable)
        glEnable(capability);
    else
        glDisable(capability);
}

GLboolean toGLBool(bool b)
{
    return b ? GL_TRUE : GL_FALSE;
}

void setStencilFace(GLenum face, const StencilFaceState& s, uint32_t reference)
{
    glStencilFuncSeparate(face, toGL(s.compareOp), static_cast<GLint>(reference), s.compareMask);
    glStencilOpSeparate(face, toGL(s.failOp), toGL(s.depthFailOp), toGL(s.passOp));
    glStencilMaskSeparate(face, s.writeMask);
}

// Sets the values of `state` that differ from `previous`; everything without a previous state.
void apply(const PipelineState& state, uint32_t reference, const PipelineState* previous, uint32_t previousReference)
{
    const bool all = previous == nullptr;

    if (all || state.primitiveRestart != previous->primitiveRestart)
    {
        if (fixedIndexRestartSupported())
            setCapability(GL_PRIMITIVE_RESTART_FIXED_INDEX, state.primitiveRestart);
        else if (state.primitiveRestart)
            SAIGA_EXIT_ERROR("primitive restart needs OpenGL 4.3 or GL_ARB_ES3_compatibility");
        // The restart with an index of its own; Vulkan has only the fixed index.
        if (all) glDisable(GL_PRIMITIVE_RESTART);
    }

    const RasterizationState& r = state.rasterization;
    if (all || r != previous->rasterization)
    {
        glPolygonMode(GL_FRONT_AND_BACK, toGL(r.polygonMode));
        setCapability(GL_CULL_FACE, r.cullMode != CullMode::None);
        glCullFace(toGL(r.cullMode));
        // With Vulkan's clip space, row 0 of the image is its top, but OpenGL takes it as the bottom: a triangle
        // that is counterclockwise in the image is clockwise for OpenGL.
        const bool ccw = (r.frontFace == FrontFace::CounterClockwise) != (cache().clipSpace == ClipSpace::Vulkan);
        glFrontFace(ccw ? GL_CCW : GL_CW);
        // Vulkan's depth bias applies to every polygon mode.
        setCapability(GL_POLYGON_OFFSET_FILL, r.depthBias);
        setCapability(GL_POLYGON_OFFSET_LINE, r.depthBias);
        setCapability(GL_POLYGON_OFFSET_POINT, r.depthBias);
        glPolygonOffset(r.depthBiasSlope, r.depthBiasConstant);
        setCapability(GL_SCISSOR_TEST, r.scissorTest);
    }

    const DepthStencilState& d = state.depthStencil;
    if (all || d != previous->depthStencil || reference != previousReference)
    {
        setCapability(GL_DEPTH_TEST, d.depthTest);
        glDepthMask(toGLBool(d.depthWrite));
        glDepthFunc(toGL(d.depthCompare));
        setCapability(GL_STENCIL_TEST, d.stencilTest);
        setStencilFace(GL_FRONT, d.front, reference);
        setStencilFace(GL_BACK, d.back, reference);
    }

    const BlendState& b = state.blend;
    if (all || b != previous->blend)
    {
        setCapability(GL_BLEND, b.enable);
        glBlendFuncSeparate(toGL(b.srcColor), toGL(b.dstColor), toGL(b.srcAlpha), toGL(b.dstAlpha));
        glBlendEquationSeparate(toGL(b.colorOp), toGL(b.alphaOp));
        glColorMask(toGLBool(b.colorWriteMask & ColorComponentR), toGLBool(b.colorWriteMask & ColorComponentG),
                    toGLBool(b.colorWriteMask & ColorComponentB), toGLBool(b.colorWriteMask & ColorComponentA));
    }
}

// Without the stencil reference: queries clamp it to the stencil bits of the bound framebuffer. Only the stencil bits
// of the masks count, as in queryStencilFace.
void validate(PipelineState expected, const char* what)
{
    for (StencilFaceState* face : {&expected.depthStencil.front, &expected.depthStencil.back})
    {
        face->compareMask &= 0xFF;
        face->writeMask &= 0xFF;
    }
    const std::string difference = pipelineStateDifference(expected, queryPipelineState());
    SAIGA_ASSERT(difference.empty(), std::string(what) + ":" + difference);
}

GLint getInt(GLenum name)
{
    GLint value = 0;
    glGetIntegerv(name, &value);
    return value;
}

GLenum getEnum(GLenum name)
{
    return static_cast<GLenum>(getInt(name));
}

const CompareOp compareOps[]     = {CompareOp::Never,          CompareOp::Less,    CompareOp::Equal,
                                    CompareOp::LessOrEqual,    CompareOp::Greater, CompareOp::NotEqual,
                                    CompareOp::GreaterOrEqual, CompareOp::Always};
const StencilOp stencilOps[]     = {StencilOp::Keep,
                                    StencilOp::Zero,
                                    StencilOp::Replace,
                                    StencilOp::IncrementAndClamp,
                                    StencilOp::DecrementAndClamp,
                                    StencilOp::Invert,
                                    StencilOp::IncrementAndWrap,
                                    StencilOp::DecrementAndWrap};
const BlendFactor blendFactors[] = {BlendFactor::Zero,     BlendFactor::One,
                                    BlendFactor::SrcColor, BlendFactor::OneMinusSrcColor,
                                    BlendFactor::DstColor, BlendFactor::OneMinusDstColor,
                                    BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha,
                                    BlendFactor::DstAlpha, BlendFactor::OneMinusDstAlpha};
const BlendOp blendOps[] = {BlendOp::Add, BlendOp::Subtract, BlendOp::ReverseSubtract, BlendOp::Min, BlendOp::Max};
const PolygonMode polygonModes[] = {PolygonMode::Fill, PolygonMode::Line, PolygonMode::Point};

StencilFaceState queryStencilFace(bool back)
{
    StencilFaceState s;
    s.compareOp   = fromGL(getEnum(back ? GL_STENCIL_BACK_FUNC : GL_STENCIL_FUNC), compareOps);
    s.compareMask = static_cast<uint32_t>(getInt(back ? GL_STENCIL_BACK_VALUE_MASK : GL_STENCIL_VALUE_MASK));
    s.writeMask   = static_cast<uint32_t>(getInt(back ? GL_STENCIL_BACK_WRITEMASK : GL_STENCIL_WRITEMASK));
    s.failOp      = fromGL(getEnum(back ? GL_STENCIL_BACK_FAIL : GL_STENCIL_FAIL), stencilOps);
    s.depthFailOp = fromGL(getEnum(back ? GL_STENCIL_BACK_PASS_DEPTH_FAIL : GL_STENCIL_PASS_DEPTH_FAIL), stencilOps);
    s.passOp      = fromGL(getEnum(back ? GL_STENCIL_BACK_PASS_DEPTH_PASS : GL_STENCIL_PASS_DEPTH_PASS), stencilOps);
    // An unsigned mask comes back as -1 through glGetIntegerv on some drivers; only the stencil bits count.
    s.compareMask &= 0xFF;
    s.writeMask &= 0xFF;
    return s;
}
}  // namespace

void applyPipelineState(const PipelineState& state, uint32_t stencilReference)
{
    Cache& c = cache();
    if (c.validation && c.enabled && c.valid) validate(c.state, "state changed outside of it");

    if (c.enabled && c.valid)
        apply(state, stencilReference, &c.state, c.stencilReference);
    else
        apply(state, stencilReference, nullptr, 0);

    c.state            = state;
    c.stencilReference = stencilReference;
    c.valid            = c.enabled;

    if (c.validation) validate(state, "applyPipelineState");
}

void clearFramebuffer(ClearBufferMask buffers, bool scissor)
{
    // Only the write masks and the scissor test matter to glClear; the rest of the state stays.
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glStencilMask(0xFF);
    setCapability(GL_SCISSOR_TEST, scissor);

    Cache& c                             = cache();
    c.state.blend.colorWriteMask         = ColorComponentAll;
    c.state.depthStencil.depthWrite      = true;
    c.state.depthStencil.front.writeMask = 0xFF;
    c.state.depthStencil.back.writeMask  = 0xFF;
    c.state.rasterization.scissorTest    = scissor;
    glClear(buffers);
}

PipelineState currentPipelineState()
{
    const Cache& c = cache();
    return c.enabled && c.valid ? c.state : queryPipelineState();
}

uint32_t currentStencilReference()
{
    const Cache& c = cache();
    return c.enabled && c.valid ? c.stencilReference : static_cast<uint32_t>(getInt(GL_STENCIL_REF));
}

PipelineState queryPipelineState(uint32_t* stencilReference)
{
    PipelineState s;
    s.primitiveRestart = fixedIndexRestartSupported() && glIsEnabled(GL_PRIMITIVE_RESTART_FIXED_INDEX) == GL_TRUE;

    RasterizationState& r = s.rasterization;
    // Some drivers return a front and a back mode.
    GLint polygonMode[2] = {};
    glGetIntegerv(GL_POLYGON_MODE, polygonMode);
    r.polygonMode = fromGL(static_cast<GLenum>(polygonMode[0]), polygonModes);
    if (glIsEnabled(GL_CULL_FACE) == GL_TRUE)
    {
        const GLenum face = getEnum(GL_CULL_FACE_MODE);
        r.cullMode = face == GL_FRONT ? CullMode::Front : face == GL_BACK ? CullMode::Back : CullMode::FrontAndBack;
    }
    const bool ccw = (getEnum(GL_FRONT_FACE) == GL_CCW) != (cache().clipSpace == ClipSpace::Vulkan);
    r.frontFace    = ccw ? FrontFace::CounterClockwise : FrontFace::Clockwise;
    r.depthBias = glIsEnabled(GL_POLYGON_OFFSET_FILL) == GL_TRUE;
    glGetFloatv(GL_POLYGON_OFFSET_FACTOR, &r.depthBiasSlope);
    glGetFloatv(GL_POLYGON_OFFSET_UNITS, &r.depthBiasConstant);
    r.scissorTest = glIsEnabled(GL_SCISSOR_TEST) == GL_TRUE;

    DepthStencilState& d = s.depthStencil;
    d.depthTest          = glIsEnabled(GL_DEPTH_TEST) == GL_TRUE;
    GLboolean depthWrite = GL_FALSE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthWrite);
    d.depthWrite   = depthWrite == GL_TRUE;
    d.depthCompare = fromGL(getEnum(GL_DEPTH_FUNC), compareOps);
    d.stencilTest  = glIsEnabled(GL_STENCIL_TEST) == GL_TRUE;
    d.front        = queryStencilFace(false);
    d.back         = queryStencilFace(true);
    if (stencilReference) *stencilReference = static_cast<uint32_t>(getInt(GL_STENCIL_REF));

    BlendState& b     = s.blend;
    b.enable          = glIsEnabled(GL_BLEND) == GL_TRUE;
    b.srcColor        = fromGL(getEnum(GL_BLEND_SRC_RGB), blendFactors);
    b.dstColor        = fromGL(getEnum(GL_BLEND_DST_RGB), blendFactors);
    b.srcAlpha        = fromGL(getEnum(GL_BLEND_SRC_ALPHA), blendFactors);
    b.dstAlpha        = fromGL(getEnum(GL_BLEND_DST_ALPHA), blendFactors);
    b.colorOp         = fromGL(getEnum(GL_BLEND_EQUATION_RGB), blendOps);
    b.alphaOp         = fromGL(getEnum(GL_BLEND_EQUATION_ALPHA), blendOps);
    GLboolean mask[4] = {};
    glGetBooleanv(GL_COLOR_WRITEMASK, mask);
    b.colorWriteMask = (mask[0] == GL_TRUE ? ColorComponentR : 0) | (mask[1] == GL_TRUE ? ColorComponentG : 0) |
                       (mask[2] == GL_TRUE ? ColorComponentB : 0) | (mask[3] == GL_TRUE ? ColorComponentA : 0);
    return s;
}

std::string pipelineStateDifference(const PipelineState& a, const PipelineState& b)
{
    std::stringstream out;
    auto check = [&](bool equal, const char* name)
    {
        if (!equal) out << " " << name;
    };
    check(a.primitiveRestart == b.primitiveRestart, "primitiveRestart");
    const RasterizationState &ra = a.rasterization, &rb = b.rasterization;
    check(ra.polygonMode == rb.polygonMode, "polygonMode");
    check(ra.cullMode == rb.cullMode, "cullMode");
    check(ra.frontFace == rb.frontFace, "frontFace");
    check(ra.depthBias == rb.depthBias, "depthBias");
    check(ra.depthBiasConstant == rb.depthBiasConstant, "depthBiasConstant");
    check(ra.depthBiasSlope == rb.depthBiasSlope, "depthBiasSlope");
    check(ra.scissorTest == rb.scissorTest, "scissorTest");
    const DepthStencilState &da = a.depthStencil, &db = b.depthStencil;
    check(da.depthTest == db.depthTest, "depthTest");
    check(da.depthWrite == db.depthWrite, "depthWrite");
    check(da.depthCompare == db.depthCompare, "depthCompare");
    check(da.stencilTest == db.stencilTest, "stencilTest");
    check(da.front == db.front, "stencil.front");
    check(da.back == db.back, "stencil.back");
    const BlendState &ba = a.blend, &bb = b.blend;
    check(ba.enable == bb.enable, "blend.enable");
    check(ba.srcColor == bb.srcColor && ba.dstColor == bb.dstColor && ba.colorOp == bb.colorOp, "blend.color");
    check(ba.srcAlpha == bb.srcAlpha && ba.dstAlpha == bb.dstAlpha && ba.alphaOp == bb.alphaOp, "blend.alpha");
    check(ba.colorWriteMask == bb.colorWriteMask, "colorWriteMask");
    return out.str();
}

void PipelineStateCache::setEnabled(bool enabled)
{
    cache().enabled = enabled;
    cache().valid   = false;
}

bool PipelineStateCache::enabled()
{
    return cache().enabled;
}

void PipelineStateCache::setValidation(bool validation)
{
    cache().validation = validation;
}

bool PipelineStateCache::validation()
{
    return cache().validation;
}

void setClipSpace(ClipSpace clipSpace)
{
    Cache& c                  = cache();
    const bool hasClipControl = getVersionMajor() * 10 + getVersionMinor() >= 45 || hasExtension("GL_ARB_clip_control");
    if (clipSpace == ClipSpace::Vulkan && !hasClipControl)
    {
        SAIGA_EXIT_ERROR("Vulkan's clip space needs OpenGL 4.5 or GL_ARB_clip_control");
    }
    // Always set: the remembered clip space may belong to another context. Without clip control, the context can only
    // have OpenGL's.
    if (hasClipControl)
    {
        // The origin stays: Vulkan's y axis points down, so the images have the same rows in memory as with Vulkan.
        glClipControl(GL_LOWER_LEFT, clipSpace == ClipSpace::Vulkan ? GL_ZERO_TO_ONE : GL_NEGATIVE_ONE_TO_ONE);
    }
    c.clipSpace = clipSpace;
    // The front face of the cached state means the other winding now.
    c.valid = false;
}

ClipSpace clipSpace()
{
    return cache().clipSpace;
}

void PipelineStateCache::invalidate()
{
    cache().valid             = false;
    cache().fixedIndexRestart = -1;
}

bool GraphicsPipeline::bind(uint32_t stencilReference)
{
    applyPipelineState(state, stencilReference);
    return shader && shader->bind();
}

void GraphicsPipeline::unbind()
{
    if (shader) shader->unbind();
}

}  // namespace Saiga
