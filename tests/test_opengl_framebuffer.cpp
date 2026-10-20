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
