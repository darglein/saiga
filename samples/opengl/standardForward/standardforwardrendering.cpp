/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "standardforwardrendering.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"

ImGui::IMConsole console;

Sample::Sample()
{
    ObjAssetLoader assetLoader;
    auto boxAsset = assetLoader.loadTexturedAsset("box.obj");

    box.asset     = boxAsset;
    box.translateGlobal(vec3(0, 2, 6));
    box.calculateModel();


    {
        const char* shaderStr = "asset/standardForwardAsset.glsl";

        auto deferredShader = shaderLoader.load<MVPTextureShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
        auto depthShader     = shaderLoader.load<MVPTextureShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEPTH", 1}});
        auto forwardShader   = shaderLoader.load<MVPTextureShader>(shaderStr);
        auto wireframeShader = shaderLoader.load<MVPTextureShader>(shaderStr);
        
        boxAsset->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
    }
    {
        const char* shaderStr = "asset/standardColoredAsset.glsl";

        auto deferredShader = shaderLoader.load<MVPColorShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
        auto depthShader     = shaderLoader.load<MVPColorShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEPTH", 1}});
        auto forwardShader   = shaderLoader.load<MVPColorShader>(shaderStr);
        auto wireframeShader = shaderLoader.load<MVPColorShader>(shaderStr);

        ((ColoredAsset*)groundPlane.asset.get())->setShader(deferredShader, forwardShader, depthShader, wireframeShader);
    }

    std::cout << "Program Initialized!" << std::endl;

    console.write("Program Initialized!\n", 20);
}



void Sample::renderOverlay(Camera* cam)
{
    Base::renderOverlay(cam);

    box.renderForward(cam);
}


void Sample::renderFinal(Camera* cam)
{
    Base::renderFinal(cam);

    console.render();
}
int main(const int argc, const char* argv[])
{
    initSaigaSample();

    {
        Sample example;

        example.run();
    }

    return 0;
}
