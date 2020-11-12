/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/RendererSampleWindow.h"
#include "saiga/opengl/world/skybox.h"


using namespace Saiga;

class Sample : public RendererSampleWindow
{
    using Base = RendererSampleWindow;

   public:
    Sample()
    {
        ObjAssetLoader assetLoader;
        auto showAsset = assetLoader.loadColoredAsset("show_model.obj");

        show.asset = showAsset;

        const char* shaderStr = renderer->getMainShaderSource();

        auto deferredShader =
            shaderLoader.load<MVPColorShaderFL>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
        auto depthShader   = shaderLoader.load<MVPColorShaderFL>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEPTH", 1}});
        auto forwardShader = shaderLoader.load<MVPColorShaderFL>(shaderStr);
        auto wireframeShader = shaderLoader.load<MVPColorShaderFL>(shaderStr);

        showAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);

        std::cout << "Program Initialized!" << std::endl;
    }



    void render(Camera* camera, RenderPass render_pass) override
    {
        Base::render(camera, render_pass);

        if (render_pass == RenderPass::Forward)
        {
            show.renderForward(camera);
        }
        else if (render_pass == RenderPass::GUI)
        {
            renderer->lighting.renderImGui();
        }
    }

   private:
    SimpleAssetObject show;
};

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
