/**
 * Copyright (c) 2021 Darius Rückert
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
    Sample()
    {
        text_atlas.loadFont("SourceSansPro-Regular.ttf", 40);

        text_overlay = TextOverlay2D(window->getWidth(), window->getHeight());

        text = std::make_shared<Text>(&text_atlas, "Saiga Text Rendering", false);
        text->params.setGlow(vec4(1, 0, 0, 1), 1);
        text->params.setColor(vec4(1, 1, 1, 1), 0.02);
        text_overlay.addText(text.get());
        text_overlay.PositionText2d(text.get(), vec2(0, 0.7), 0.2);


        counter_text = std::make_shared<Text>(&text_atlas, "test", false);
        counter_text->params.setColor(vec4(0, 0, 0, 1), 0.02);
        counter_text->params.setOutline(vec4(1, 1, 1, 1), 0.05, 0.02);
        text_overlay.addText(counter_text.get());
        text_overlay.PositionText2d(counter_text.get(), vec2(0.1, 0.4), 0.2);

        std::cout << "Program Initialized!" << std::endl;
    }

    void update(float dt) override
    {
        Base::update(dt);
        counter_text->updateText("Frame " + std::to_string(count));
        count++;
    }

    void render(RenderInfo render_info) override
    {
        if (render_info.render_pass == RenderPass::GUI)
        {
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            renderer->bindCamera(text_overlay.GetCamera());
            text_overlay.render();
        }
    }



   private:
    std::shared_ptr<Text> text;
    std::shared_ptr<Text> counter_text;
    TextOverlay2D text_overlay;
    TextAtlas text_atlas;
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
