/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/shader/shader.h"
#include "saiga/opengl/shaderStorageBuffer.h"
#include "saiga/opengl/texture/Texture2D.h"

#include "opengl_test_context.h"

using namespace Saiga;

// A triangle covering the whole viewport, from gl_VertexID (no vertex buffer needed).
static const char* fullscreen_color_shader = R"(
##GL_VERTEX_SHADER
#version 330
void main()
{
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
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

// out[i] = in[i] * in[i] + offset
static const char* square_compute_shader = R"(
##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer In { uint data_in[]; };
layout(std430, binding = 1) writeonly buffer Out { uint data_out[]; };
uniform uint offset;
uniform uint n;
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    data_out[i] = data_in[i] * data_in[i] + offset;
}
)";

// Writes the pixel coordinate into an rgba8ui image.
static const char* image_compute_shader = R"(
##GL_COMPUTE_SHADER
#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 2, rgba8ui) uniform writeonly uimage2D dst;
void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    imageStore(dst, p, uvec4(p.x, p.y, p.x + p.y, 255));
}
)";

class OpenGLShader : public OpenGLTest
{
};

TEST_F(OpenGLShader, CompileError)
{
    Shader shader;
    shader.init_from_string("##GL_FRAGMENT_SHADER\n#version 330\nvoid main() { this is not glsl }\n");
    EXPECT_EQ(shader.program, 0u);
    EXPECT_FALSE(shader.bind());
}

// init returned true on a link error.
TEST_F(OpenGLShader, LinkErrorReturnsFalse)
{
    Shader shader;
    // Compiles, but the call of the undefined function f fails to link.
    EXPECT_FALSE(shader.init_from_string(
        "##GL_FRAGMENT_SHADER\n#version 330\nvec4 f();\nout vec4 c;\nvoid main() { c = f(); }\n"));
    EXPECT_EQ(shader.program, 0u);

    EXPECT_TRUE(shader.init_from_string(fullscreen_color_shader));
    EXPECT_NE(shader.program, 0u);
}

// A second init (as on a reload) leaked the GL program of the first one.
TEST_F(OpenGLShader, ReinitDeletesOldProgram)
{
    Shader shader;
    ASSERT_TRUE(shader.init_from_string(fullscreen_color_shader));
    GLuint first = shader.program;
    ASSERT_NE(first, 0u);

    ASSERT_TRUE(shader.init_from_string(fullscreen_color_shader));
    EXPECT_NE(shader.program, 0u);
    // The driver may reuse the name of the deleted program.
    EXPECT_TRUE(shader.program == first || glIsProgram(first) == GL_FALSE);
}

// A reinit of a bound shader left the deleted program bound and Shader::boundShader pointing to it.
TEST_F(OpenGLShader, ReinitWhileBound)
{
    Shader shader;
    ASSERT_TRUE(shader.init_from_string(fullscreen_color_shader));
    ASSERT_TRUE(shader.bind());

    ASSERT_TRUE(shader.init_from_string(fullscreen_color_shader));
    EXPECT_EQ(Shader::getBoundShader(), 0u);
    GLint current = -1;
    glGetIntegerv(GL_CURRENT_PROGRAM, &current);
    EXPECT_EQ(current, 0);

    ASSERT_TRUE(shader.bind());
    EXPECT_TRUE(shader.isBound());
    shader.unbind();
    EXPECT_EQ(Shader::getBoundShader(), 0u);
}

TEST_F(OpenGLShader, RenderFullscreenTriangle)
{
    int w = 16, h = 12;

    Shader shader;
    shader.init_from_string(fullscreen_color_shader);
    ASSERT_NE(shader.program, 0u);
    GLint color_location = shader.getUniformLocation("color");
    ASSERT_GE(color_location, 0);

    auto tex = std::make_shared<Texture>();
    tex->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
    Framebuffer fb;
    fb.create();
    fb.attachTexture(tex);
    fb.drawToAll();
    fb.check();

    glViewport(0, 0, w, h);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);

    // The core profile needs a vertex array object for a draw call.
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    ASSERT_TRUE(shader.bind());
    shader.upload(color_location, vec4(1.f, 0.5f, 0.f, 1.f));
    glDrawArrays(GL_TRIANGLES, 0, 3);
    shader.unbind();

    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);

    auto img = ReadPixels(w, h);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            auto c = img(y, x);
            EXPECT_EQ(c(0), 255);
            EXPECT_NEAR(c(1), 128, 1);
            EXPECT_EQ(c(2), 0);
            EXPECT_EQ(c(3), 255);
        }
    }

    fb.unbind();
}

TEST_F(OpenGLShader, ComputeShaderBuffer)
{
    int n = 1000;
    std::vector<unsigned int> data(n);
    for (int i = 0; i < n; ++i) data[i] = i;

    std::vector<unsigned int> zeros(n, 0);
    TemplatedShaderStorageBuffer<unsigned int> in_buffer, out_buffer;
    in_buffer.create(data, GL_STATIC_DRAW);
    out_buffer.create(zeros, GL_DYNAMIC_READ);

    Shader shader;
    shader.init_from_string(square_compute_shader);
    ASSERT_NE(shader.program, 0u);
    EXPECT_EQ(shader.getComputeWorkGroupSize(), uvec3(64, 1, 1));

    in_buffer.bind(0);
    out_buffer.bind(1);
    ASSERT_TRUE(shader.bind());
    shader.upload(shader.getUniformLocation("offset"), 7u);
    shader.upload(shader.getUniformLocation("n"), static_cast<unsigned int>(n));
    auto groups = shader.getNumGroupsCeil(uvec3(n, 1, 1));
    EXPECT_EQ(groups, uvec3(16, 1, 1));
    shader.dispatchCompute(groups);
    shader.unbind();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<unsigned int> result(n);
    out_buffer.get(result);
    for (int i = 0; i < n; ++i) EXPECT_EQ(result[i], static_cast<unsigned int>(i * i + 7)) << i;
}

TEST_F(OpenGLShader, ComputeShaderImage)
{
    int w = 20, h = 13;
    Texture2D tex;
    tex.create(w, h, GL_RGBA_INTEGER, GL_RGBA8UI, GL_UNSIGNED_BYTE);

    Shader shader;
    shader.init_from_string(image_compute_shader);
    ASSERT_NE(shader.program, 0u);

    tex.bindImageTexture(2, GL_WRITE_ONLY);
    ASSERT_TRUE(shader.bind());
    shader.dispatchCompute(shader.getNumGroupsCeil(uvec3(w, h, 1)));
    shader.unbind();
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);

    TemplatedImage<ucvec4> img(h, w);
    ASSERT_TRUE(tex.download(img.data()));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) EXPECT_EQ(img(y, x), ucvec4(x, y, x + y, 255)) << x << " " << y;
}
