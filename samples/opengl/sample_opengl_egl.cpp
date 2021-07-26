/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/rendering/forwardRendering/forward_renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"

using namespace Saiga;

class Sample : public StandaloneWindow<WindowManagement::EGL, ForwardRenderer>
{
   public:
    PerspectiveCamera camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject sphere;

    ProceduralSkybox skybox;


    Sample() : StandaloneWindow("config.ini")
    {
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

    void update(float dt) override {}
    void interpolate(float dt, float interpolation) override {}

    void render(RenderInfo render_info) override
    {
        skybox.render(render_info.camera);
        groundPlane.renderForward(render_info.camera);
        cube1.renderForward(render_info.camera);
        cube2.renderForward(render_info.camera);
        sphere.renderForward(render_info.camera);


        window->ScreenshotDefaultFramebuffer().save("render.png");
        window->close();
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
