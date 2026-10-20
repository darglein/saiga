/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/texture/Texture2D.h"

#include "opengl_test_context.h"

using namespace Saiga;

class OpenGLFramebuffer : public OpenGLTest
{
   protected:
    static std::shared_ptr<Texture> ColorTexture(int w, int h)
    {
        auto tex = std::make_shared<Texture>();
        tex->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
        return tex;
    }

    static std::shared_ptr<Texture> DepthTexture(int w, int h)
    {
        auto tex = std::make_shared<Texture>();
        tex->create(w, h, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F, GL_FLOAT);
        return tex;
    }
};

// A second create() leaked the first framebuffer object.
TEST_F(OpenGLFramebuffer, RecreateDeletesOldFramebuffer)
{
    Framebuffer fb;
    fb.create();
    GLuint first = fb.getId();
    ASSERT_NE(first, 0u);

    fb.create();
    EXPECT_NE(fb.getId(), 0u);
    // The driver may reuse the name of the deleted framebuffer.
    EXPECT_TRUE(fb.getId() == first || glIsFramebuffer(first) == GL_FALSE);
    fb.unbind();
}

// drawTo with no ids indexed into an empty vector.
TEST_F(OpenGLFramebuffer, DrawToNoBuffers)
{
    Framebuffer fb;
    fb.create();
    fb.attachTexture(ColorTexture(4, 4));
    fb.drawTo({});
    GLint draw_buffer = -1;
    glGetIntegerv(GL_DRAW_BUFFER0, &draw_buffer);
    EXPECT_EQ(draw_buffer, static_cast<GLint>(GL_NONE));
    fb.unbind();
}

TEST_F(OpenGLFramebuffer, ClearColorAndDepth)
{
    int w = 32, h = 16;
    Framebuffer fb;
    fb.create();
    fb.attachTexture(ColorTexture(w, h));
    fb.attachTextureDepth(DepthTexture(w, h));
    fb.drawToAll();
    fb.check();

    glViewport(0, 0, w, h);
    glClearColor(0.f, 1.f, 0.f, 1.f);
    glClearDepth(0.25);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto img = ReadPixels(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) EXPECT_EQ(img(y, x), ucvec4(0, 255, 0, 255)) << x << " " << y;

    // The attached textures hold the result too
    TemplatedImage<ucvec4> color(h, w);
    ASSERT_TRUE(fb.getTextureColor(0)->download(color.data()));
    EXPECT_EQ(color(h / 2, w / 2), ucvec4(0, 255, 0, 255));

    std::vector<float> depth(w * h, 0);
    ASSERT_TRUE(fb.getTextureDepth()->download(depth.data()));
    for (auto d : depth) EXPECT_FLOAT_EQ(d, 0.25f);

    fb.unbind();
}

TEST_F(OpenGLFramebuffer, MultipleRenderTargets)
{
    int w = 8, h = 8;
    Framebuffer fb;
    fb.create();
    fb.attachTexture(ColorTexture(w, h));
    fb.attachTexture(ColorTexture(w, h));
    fb.drawToAll();
    fb.check();

    float red[4]  = {1, 0, 0, 1};
    float blue[4] = {0, 0, 1, 1};
    glClearBufferfv(GL_COLOR, 0, red);
    glClearBufferfv(GL_COLOR, 1, blue);

    TemplatedImage<ucvec4> img0(h, w), img1(h, w);
    ASSERT_TRUE(fb.getTextureColor(0)->download(img0.data()));
    ASSERT_TRUE(fb.getTextureColor(1)->download(img1.data()));
    EXPECT_EQ(img0(3, 4), ucvec4(255, 0, 0, 255));
    EXPECT_EQ(img1(3, 4), ucvec4(0, 0, 255, 255));

    // Draw only to the second buffer
    fb.drawTo({1});
    glClearColor(1.f, 1.f, 1.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ASSERT_TRUE(fb.getTextureColor(0)->download(img0.data()));
    ASSERT_TRUE(fb.getTextureColor(1)->download(img1.data()));
    EXPECT_EQ(img0(3, 4), ucvec4(255, 0, 0, 255));
    EXPECT_EQ(img1(3, 4), ucvec4(255, 255, 255, 255));

    fb.unbind();
}

TEST_F(OpenGLFramebuffer, Resize)
{
    Framebuffer fb;
    fb.create();
    fb.attachTexture(ColorTexture(4, 4));
    fb.attachTextureDepth(DepthTexture(4, 4));
    fb.check();

    fb.resize(20, 10);
    EXPECT_EQ(fb.getTextureColor(0)->getWidth(), 20);
    EXPECT_EQ(fb.getTextureColor(0)->getHeight(), 10);
    EXPECT_EQ(fb.getTextureDepth()->getWidth(), 20);
    fb.check();

    glViewport(0, 0, 20, 10);
    glClearColor(0.f, 0.f, 1.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    auto img = ReadPixels(20, 10);
    EXPECT_EQ(img(9, 19), ucvec4(0, 0, 255, 255));

    fb.unbind();
}

TEST_F(OpenGLFramebuffer, BlitColor)
{
    int w = 6, h = 6;
    Framebuffer src, dst;
    src.create();
    src.attachTexture(ColorTexture(w, h));
    src.check();
    glClearColor(1.f, 0.f, 1.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    dst.create();
    dst.attachTexture(ColorTexture(w, h));
    dst.check();
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);

    src.blitColor(dst.getId());

    TemplatedImage<ucvec4> img(h, w);
    ASSERT_TRUE(dst.getTextureColor(0)->download(img.data()));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) EXPECT_EQ(img(y, x), ucvec4(255, 0, 255, 255)) << x << " " << y;

    Framebuffer::bindDefaultFramebuffer();
}

TEST_F(OpenGLFramebuffer, BlitDepth)
{
    int w = 5, h = 4;
    Framebuffer src, dst;
    src.create();
    src.attachTextureDepth(DepthTexture(w, h));
    src.drawToNone();
    src.check();
    glClearDepth(0.25);
    glClear(GL_DEPTH_BUFFER_BIT);

    dst.create();
    dst.attachTextureDepth(DepthTexture(w, h));
    dst.drawToNone();
    dst.check();
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);

    src.blitDepth(dst.getId());

    std::vector<float> depth(w * h, 0.f);
    ASSERT_TRUE(dst.getTextureDepth()->download(depth.data()));
    for (float d : depth) EXPECT_FLOAT_EQ(d, 0.25f);

    Framebuffer::bindDefaultFramebuffer();
}

TEST_F(OpenGLFramebuffer, DepthStencilAttachment)
{
    int w = 4, h = 3;
    auto depth_stencil = std::make_shared<Texture>();
    depth_stencil->create(w, h, GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8);

    Framebuffer fb;
    fb.create();
    fb.attachTexture(ColorTexture(w, h));
    fb.attachTextureDepthStencil(depth_stencil);
    fb.check();
    EXPECT_EQ(fb.getTextureDepth(), depth_stencil);
    EXPECT_EQ(fb.getTextureStencil(), depth_stencil);

    fb.resize(7, 6);
    EXPECT_EQ(depth_stencil->getWidth(), 7);
    EXPECT_EQ(depth_stencil->getHeight(), 6);
    fb.check();

    glClearStencil(0x5a);
    glClear(GL_STENCIL_BUFFER_BIT);
    std::vector<unsigned char> stencil(7 * 6, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, 7, 6, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, stencil.data());
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    for (auto s : stencil) EXPECT_EQ(s, 0x5a);

    fb.unbind();
}
