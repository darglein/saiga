/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
// applyPipelineState, clearFramebuffer, the PipelineStateCache and GraphicsPipeline on a headless EGL context.
// On Mesa's software renderer, as CI does:
//   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json ctest -R test_opengl_pipeline

#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/pipeline.h"
#include "saiga/opengl/shader/shader.h"
#include "saiga/opengl/texture/Texture2D.h"

#include "opengl_test_context.h"

using namespace Saiga;

namespace
{
// A triangle covering the whole viewport at the depth `depth`, from gl_VertexID.
const char* fullscreen_shader = R"(
##GL_VERTEX_SHADER
#version 330
uniform float depth;
void main()
{
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, depth, 1.0);
}

##GL_FRAGMENT_SHADER
#version 330
uniform vec4 color;
layout(location = 0) out vec4 out_color;
void main()
{
    out_color = color;
}
)";

// Every field differs from the defaults.
// Fixed-index primitive restart is core since OpenGL 4.3; the context may be 3.3.
bool FixedIndexRestartSupported()
{
    return getVersionMajor() * 10 + getVersionMinor() >= 43 || hasExtension("GL_ARB_ES3_compatibility");
}

PipelineState NonDefaultState()
{
    PipelineState s;
    s.primitiveRestart = FixedIndexRestartSupported();

    RasterizationState& r = s.rasterization;
    r.polygonMode         = PolygonMode::Line;
    r.cullMode            = CullMode::Front;
    r.frontFace           = FrontFace::Clockwise;
    r.depthBias           = true;
    r.depthBiasConstant   = 2.f;
    r.depthBiasSlope      = 1.5f;
    r.scissorTest         = true;

    DepthStencilState& d = s.depthStencil;
    d.depthTest          = true;
    d.depthWrite         = true;
    d.depthCompare       = CompareOp::GreaterOrEqual;
    d.stencilTest        = true;
    d.front.failOp       = StencilOp::Zero;
    d.front.passOp       = StencilOp::Replace;
    d.front.depthFailOp  = StencilOp::IncrementAndWrap;
    d.front.compareOp    = CompareOp::NotEqual;
    d.front.compareMask  = 0x0F;
    d.front.writeMask    = 0xF0;
    d.back.failOp        = StencilOp::Invert;
    d.back.passOp        = StencilOp::DecrementAndClamp;
    d.back.depthFailOp   = StencilOp::IncrementAndClamp;
    d.back.compareOp     = CompareOp::Greater;
    d.back.compareMask   = 0x3C;
    d.back.writeMask     = 0x81;

    BlendState& b    = s.blend;
    b.enable         = true;
    b.srcColor       = BlendFactor::DstColor;
    b.dstColor       = BlendFactor::OneMinusSrcColor;
    b.colorOp        = BlendOp::ReverseSubtract;
    b.srcAlpha       = BlendFactor::SrcAlpha;
    b.dstAlpha       = BlendFactor::OneMinusDstAlpha;
    b.alphaOp        = BlendOp::Max;
    b.colorWriteMask = ColorComponentR | ColorComponentB;
    return s;
}

// State that a careless pass could leave behind, set with gl calls directly.
void SetUnrelatedState()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    glEnable(GL_SCISSOR_TEST);
    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_NEVER, 3, 0x0F);
    glDepthMask(GL_FALSE);
    glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(4.f, 4.f);
}

class OpenGLPipeline : public OpenGLTest
{
   public:
    static ivec4 Center(int w, int h) { return ReadPixels(w, h)(h / 2, w / 2).cast<int>(); }

   protected:
    void TearDown() override
    {
        PipelineStateCache::setEnabled(false);
        PipelineStateCache::setValidation(false);
        if (context_available) applyPipelineState(PipelineState());
        OpenGLTest::TearDown();
    }
};

// A color and depth-stencil target with a VAO for drawing the fullscreen triangle.
struct Target
{
    Target(int w, int h) : w(w), h(h)
    {
        color = std::make_shared<Texture>();
        color->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
        depthStencil = std::make_shared<Texture>();
        depthStencil->create(w, h, GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8);
        fb.create();
        fb.attachTexture(color);
        fb.attachTextureDepthStencil(depthStencil);
        fb.drawToAll();
        fb.check();
        fb.bind();
        glViewport(0, 0, w, h);
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }
    ~Target()
    {
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &vao);
        fb.unbind();
    }

    void Draw(GraphicsPipeline& pipeline, vec4 c, float depth = 0.f, uint32_t stencilReference = 0)
    {
        ASSERT_TRUE(pipeline.bind(stencilReference));
        pipeline.shader->upload(pipeline.shader->getUniformLocation("color"), c);
        pipeline.shader->upload(pipeline.shader->getUniformLocation("depth"), depth);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        pipeline.unbind();
    }

    ivec4 Pixel() { return OpenGLPipeline::Center(w, h); }

    int w, h;
    std::shared_ptr<Texture> color, depthStencil;
    Framebuffer fb;
    GLuint vao = 0;
};

std::shared_ptr<Shader> FullscreenShader()
{
    auto shader = std::make_shared<Shader>();
    shader->init_from_string(fullscreen_shader);
    return shader;
}
}  // namespace

TEST_F(OpenGLPipeline, SetsEveryField)
{
    const PipelineState state = NonDefaultState();
    applyPipelineState(state, 0x5A);
    uint32_t reference = 0;
    EXPECT_EQ(pipelineStateDifference(state, queryPipelineState(&reference)), "");
    EXPECT_EQ(reference, 0x5Au);

    applyPipelineState(PipelineState());
    EXPECT_EQ(pipelineStateDifference(PipelineState(), queryPipelineState()), "");
}

TEST_F(OpenGLPipeline, IndependentOfEarlierState)
{
    SetUnrelatedState();
    applyPipelineState(PipelineState::opaque());
    EXPECT_EQ(pipelineStateDifference(PipelineState::opaque(), queryPipelineState()), "");
}

TEST_F(OpenGLPipeline, DifferenceNamesTheFields)
{
    PipelineState other          = PipelineState();
    other.blend                  = BlendState::additive();
    other.rasterization.cullMode = CullMode::Back;
    EXPECT_EQ(pipelineStateDifference(PipelineState(), other), " cullMode blend.enable blend.color blend.alpha");
}

TEST_F(OpenGLPipeline, CacheSkipsOnlyUnchangedValues)
{
    PipelineStateCache::setEnabled(true);
    PipelineStateCache::setValidation(true);

    PipelineState a                = NonDefaultState();
    PipelineState b                = a;
    b.blend                        = BlendState::alpha();
    b.depthStencil.front.compareOp = CompareOp::Always;

    for (const PipelineState* s : {&a, &b, &b, &a, &b})
    {
        applyPipelineState(*s, 7);
        EXPECT_EQ(pipelineStateDifference(*s, queryPipelineState()), "");
    }

    // Only the stencil reference changes.
    applyPipelineState(b, 9);
    uint32_t reference = 0;
    queryPipelineState(&reference);
    EXPECT_EQ(reference, 9u);
}

TEST_F(OpenGLPipeline, InvalidateAfterDirectChanges)
{
    PipelineStateCache::setEnabled(true);
    applyPipelineState(PipelineState());
    SetUnrelatedState();
    PipelineStateCache::invalidate();
    applyPipelineState(PipelineState());
    EXPECT_EQ(pipelineStateDifference(PipelineState(), queryPipelineState()), "");
}

// Without the cache, the current state is read back, also what gl calls changed directly.
TEST_F(OpenGLPipeline, CurrentStateHasDirectChanges)
{
    applyPipelineState(PipelineState::opaque(), 3);
    glEnable(GL_BLEND);
    PipelineState expected = PipelineState::opaque();
    expected.blend.enable  = true;
    EXPECT_EQ(pipelineStateDifference(expected, currentPipelineState()), "");

    PipelineStateCache::setEnabled(true);
    applyPipelineState(PipelineState::opaque(), 3);
    EXPECT_EQ(pipelineStateDifference(PipelineState::opaque(), currentPipelineState()), "");
    EXPECT_EQ(currentStencilReference(), 3u);
}

// A mask with more bits than the stencil buffer has is valid; the validation only compares the stencil bits.
TEST_F(OpenGLPipeline, ValidationIgnoresMaskBitsBeyondTheStencil)
{
    PipelineStateCache::setValidation(true);
    PipelineState s            = PipelineState();
    s.depthStencil.stencilTest = true;
    StencilFaceState face;
    face.compareMask = 0xFFFFFFFF;
    face.writeMask   = 0xFFFFFFFF;
    s.depthStencil.setStencil(face);
    applyPipelineState(s);
}

TEST_F(OpenGLPipeline, ClearIgnoresTheWriteMasks)
{
    Target target(8, 8);
    PipelineState masked;
    masked.blend.colorWriteMask         = 0;
    masked.depthStencil.front.writeMask = 0;
    masked.depthStencil.back.writeMask  = 0;
    masked.rasterization.scissorTest    = true;
    applyPipelineState(masked);
    glScissor(0, 0, 1, 1);

    glClearColor(0.f, 1.f, 0.f, 1.f);
    clearFramebuffer(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    EXPECT_EQ(target.Pixel(), ivec4(0, 255, 0, 255));
}

TEST_F(OpenGLPipeline, CacheFollowsClear)
{
    PipelineStateCache::setEnabled(true);
    PipelineStateCache::setValidation(true);
    Target target(8, 8);
    PipelineState masked;
    masked.blend.colorWriteMask = 0;
    applyPipelineState(masked);
    clearFramebuffer(GL_COLOR_BUFFER_BIT);
    // The validation would assert if the cache still had the color writes off.
    applyPipelineState(masked);
    EXPECT_EQ(pipelineStateDifference(masked, queryPipelineState()), "");
}

TEST_F(OpenGLPipeline, BlendsAfterCarelessState)
{
    Target target(8, 8);
    auto shader = FullscreenShader();

    GraphicsPipeline background(shader, PipelineState());
    PipelineState blended = PipelineState();
    blended.blend         = BlendState::alpha();
    GraphicsPipeline over(shader, blended);

    SetUnrelatedState();
    target.Draw(background, vec4(1.f, 0.f, 0.f, 1.f));
    target.Draw(over, vec4(0.f, 0.f, 1.f, 0.5f));
    const ivec4 pixel = target.Pixel();
    EXPECT_NEAR(pixel(0), 128, 1);
    EXPECT_EQ(pixel(1), 0);
    EXPECT_NEAR(pixel(2), 128, 1);
    EXPECT_EQ(pixel(3), 255);
}

TEST_F(OpenGLPipeline, DepthAndStencilTests)
{
    Target target(8, 8);
    auto shader = FullscreenShader();
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClearDepth(1.0);
    glClearStencil(0);
    clearFramebuffer(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // Writes depth 0.5 (NDC 0) and stencil 1.
    PipelineState mark            = PipelineState::opaque();
    mark.depthStencil.stencilTest = true;
    StencilFaceState replace;
    replace.passOp = StencilOp::Replace;
    mark.depthStencil.setStencil(replace);
    GraphicsPipeline marker(shader, mark);
    target.Draw(marker, vec4(1.f, 0.f, 0.f, 1.f), 0.f, 1);

    // Behind the marker: fails the depth test.
    GraphicsPipeline behind(shader, PipelineState::opaque());
    target.Draw(behind, vec4(0.f, 1.f, 0.f, 1.f), 0.5f);
    EXPECT_EQ(target.Pixel(), ivec4(255, 0, 0, 255));

    // In front, but only where the stencil isn't 1: fails the stencil test.
    PipelineState notMarked            = PipelineState::opaque();
    notMarked.depthStencil.stencilTest = true;
    StencilFaceState notEqual;
    notEqual.compareOp = CompareOp::NotEqual;
    notMarked.depthStencil.setStencil(notEqual);
    GraphicsPipeline outside(shader, notMarked);
    target.Draw(outside, vec4(0.f, 0.f, 1.f, 1.f), -0.5f, 1);
    EXPECT_EQ(target.Pixel(), ivec4(255, 0, 0, 255));

    // The same with reference 2 passes.
    target.Draw(outside, vec4(0.f, 0.f, 1.f, 1.f), -0.5f, 2);
    EXPECT_EQ(target.Pixel(), ivec4(0, 0, 255, 255));
}

TEST_F(OpenGLPipeline, CullsFaces)
{
    Target target(8, 8);
    auto shader = FullscreenShader();
    glClearColor(0.f, 0.f, 0.f, 1.f);
    clearFramebuffer(GL_COLOR_BUFFER_BIT);

    // The fullscreen triangle is counter-clockwise.
    PipelineState cullFront          = PipelineState();
    cullFront.rasterization.cullMode = CullMode::Front;
    GraphicsPipeline culled(shader, cullFront);
    target.Draw(culled, vec4(1.f, 1.f, 1.f, 1.f));
    EXPECT_EQ(target.Pixel(), ivec4(0, 0, 0, 255));

    cullFront.rasterization.frontFace = FrontFace::Clockwise;
    GraphicsPipeline drawn(shader, cullFront);
    target.Draw(drawn, vec4(1.f, 1.f, 1.f, 1.f));
    EXPECT_EQ(target.Pixel(), ivec4(255, 255, 255, 255));
}
