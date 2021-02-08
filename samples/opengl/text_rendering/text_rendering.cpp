/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/text/all.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"
using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    Sample() : layout(window->getWidth(), window->getHeight())
    {
        // This simple AssetLoader can create assets from meshes and generate some generic debug assets
        ObjAssetLoader assetLoader;
        teapot.asset = assetLoader.loadColoredAsset("models/teapot.obj");
        teapot.translateGlobal(vec3(0, 1, 0));
        teapot.calculateModel();


        text_atlas.loadFont("SourceSansPro-Regular.ttf");
        text_overlay = TextOverlay2D(window->getWidth(), window->getHeight());

        text = std::make_shared<Text>(&text_atlas, "test", false);
        text_overlay.addText(text.get());

        AABB bb = text->getAabb();
        vec2 relPos(0.5, 0.5);
        layout.transform(text.get(), bb, relPos, 0.1, Layout::LEFT, Layout::RIGHT);

        std::cout << "Program Initialized!" << std::endl;
    }

    void update(float dt) override
    {
        Base::update(dt);
        text->updateText(std::to_string(count));
        count++;
    }

    void render(Camera* cam, RenderPass render_pass) override
    {
        if (render_pass == RenderPass::GUI)
        {
            renderer->bindCamera(&layout.cam);
            text_overlay.render();
        }
    }



   private:
    SimpleAssetObject teapot;

    std::shared_ptr<Text> text;
    TextOverlay2D text_overlay;
    Layout layout;
    TextureAtlas text_atlas;
    int count = 0;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    Sample window;
    window.run();
    return 0;
}
