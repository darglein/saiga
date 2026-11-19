/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/rendering/forwardRendering/forward_renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"
#include "saiga/opengl/window/StandaloneWindow.h"

using namespace Saiga;

static WindowParameters HeadlessWindowParameters()
{
    WindowParameters params;
    params.fromConfigFile("config.ini");
    params.backend = WindowParameters::Backend::Headless;
    // With ImGui, the editor layout would size the 3D view by its ImGui window instead of the window size.
    params.imguiParameters.enable = false;
    return params;
}

static OpenGLParameters OpenGLParametersFromConfig()
{
    OpenGLParameters params;
    params.fromConfigFile("config.ini");
    return params;
}

class Sample : public StandaloneWindow
{
   public:
    ForwardRenderer* renderer;
    PerspectiveCamera camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;
    bool rendered = false;


    Sample() : StandaloneWindow(HeadlessWindowParameters(), OpenGLParametersFromConfig())
    {
        // The renderer and the main loop read their parameters from it as well.
        config_file = "config.ini";
        renderer = CreateRenderer<ForwardRenderer>();

        // create a perspective camera
        float aspect = window->getAspectRatio();
        camera.setProj(60.0f, aspect, 0.1f, 50.0f);
        camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));

        // Set the camera from which view the scene is rendered
        window->setCamera(&camera);

        auto cubeAsset = std::make_shared<ColoredAsset>(
            BoxMesh(AABB(vec3(-1, -1, -1), vec3(1, 1, 1))).SetVertexColor(vec4(0.7, 0.7, 0.7, 1)));

        // Rendering an asset at a user defined location is done most efficiently with a 4x4 transformation matrix,
        // that is passed to the shader as a uniform. The SimpleAssetObject does exactly this. It contains a
        // transformation matrix and simple transformation methods for example 'translate' 'rotate'. The 'render'
        // methods of a SimpleAssetObject will bind the correct shaders, upload the matrix to the correct uniform and
        // call the raw 'render' of the referenced asset.
        cube1.asset = cubeAsset;

        // An asset can be referenced by multiple SimpleAssetObject, because each SimpleAssetObject has its own
        // transformation matrix and therefore they all can be drawn at different locations.
        cube2.asset = cubeAsset;

        // Translate the first cube
        cube1.translateGlobal(vec3(3, 1, 0));
        // Compute the 4x4 transformation matrix. This has to be done before rendering when a 'transform method' was
        // called.
        cube1.calculateModel();

        cube2.translateGlobal(vec3(3, 1, 5));
        cube2.calculateModel();


        //        auto sphereMesh  = TriangleMeshGenerator::IcoSphereMesh(Sphere(make_vec3(0), 1), 2);
        //        auto sphereAsset = assetLoader.assetFromMesh(*sphereMesh, Colors::green);

        auto sphereAsset = std::make_shared<ColoredAsset>(
            IcoSphereMesh(Sphere(make_vec3(0), 1), 2).SetVertexColor(vec4(0.7, 0.7, 0.7, 1)));
        sphere.asset = sphereAsset;
        sphere.translateGlobal(vec3(-2, 1, 0));
        sphere.calculateModel();

        groundPlane.asset = std::make_shared<ColoredAsset>(
            CheckerBoardPlane(make_ivec2(20, 20), 1.0f, Colors::firebrick, Colors::gray));


        std::cout << "Program Initialized!" << std::endl;
    }
    ~Sample()
    {
        // We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
    }

    void update(float dt) override
    {
        if (!rendered) return;
        // The renderer draws the scene into its target framebuffer (the 3D view of the editor layout).
        auto color = renderer->target_framebuffer->getTextureColor(0);
        TemplatedImage<ucvec4> img(color->getHeight(), color->getWidth());
        color->download(img.data8());
        img.getImageView().flipY();
        img.save("render.png");
        window->close();
    }
    void interpolate(float dt, float interpolation) override {}

    void render(RenderInfo render_info) override
    {
        skybox.render(render_info.camera);
        groundPlane.renderForward(render_info.camera);
        cube1.renderForward(render_info.camera);
        cube2.renderForward(render_info.camera);
        sphere.renderForward(render_info.camera);

        rendered = true;
    }
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
