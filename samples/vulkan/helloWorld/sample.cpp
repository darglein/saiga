/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "sample.h"

#include "saiga/core/image/imageTransformations.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/random.h"

#include <saiga/core/imgui/imgui.h>

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);


    //    cout << camera.view << endl;
    //    cout << camera.proj << endl;
    //    exit(0);

    window.setCamera(&camera);
}

VulkanExample::~VulkanExample()
{
    assetRenderer.destroy();
    lineAssetRenderer.destroy();
    pointCloudRenderer.destroy();
    texturedAssetRenderer.destroy();
}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    {
        auto tex = std::make_shared<Saiga::Vulkan::Texture2D>();

        Saiga::Image img("box.png");

        cout << "uncompressed size " << img.size() << endl;
        auto data = img.compress();
        cout << "compressed size " << data.size() << endl;
        img.decompress(data);
        cout << "test" << endl;

        if (img.type == Saiga::UC3)
        {
            cout << "adding alplha channel" << endl;
            Saiga::TemplatedImage<ucvec4> img2(img.height, img.width);
            cout << img << " " << img2 << endl;
            Saiga::ImageTransformation::addAlphaChannel(img.getImageView<ucvec3>(), img2.getImageView(), 255);
            tex->fromImage(base, img2);
        }
        else
        {
            cout << img << endl;
            tex->fromImage(base, img);
        }
        texture = tex;
    }


    assetRenderer.init(base, renderer.renderPass);
    lineAssetRenderer.init(base, renderer.renderPass, 2);
    pointCloudRenderer.init(base, renderer.renderPass, 5);
    texturedAssetRenderer.init(base, renderer.renderPass);
    textureDisplay.init(base, renderer.renderPass);

    textureDes = textureDisplay.createAndUpdateDescriptorSet(*texture);


    box.loadObj("box.obj");
    //    box.loadObj("cat.obj");
    box.init(renderer.base());
    box.descriptor = texturedAssetRenderer.createAndUpdateDescriptorSet(*box.textures[0]);

    //    teapot.loadObj("teapot.obj");
    teapot.loadObj("teapot.obj");
    teapot.init(renderer.base());
    teapotTrans.translateGlobal(vec3(0, 1, 0));
    teapotTrans.calculateModel();

    plane.createCheckerBoard(ivec2(20, 20), 1.0f, Saiga::Colors::firebrick, Saiga::Colors::gray);
    plane.init(renderer.base());

    grid.createGrid(10, 10);
    grid.init(renderer.base());

    frustum.createFrustum(camera.proj, 2, make_vec4(1), true);
    frustum.init(renderer.base());

    pointCloud.init(base, 1000 * 1000);
    for (int i = 0; i < 1000 * 1000; ++i)
    {
        Saiga::VertexNC v;
        v.position               = make_vec4(linearRand(make_vec3(-3), make_vec3(3)), 1);
        v.color                  = make_vec4(linearRand(make_vec3(0), make_vec3(1)), 1);
        pointCloud.pointCloud[i] = v;
    }
}



void VulkanExample::update(float dt)
{
    camera.update(dt);
    camera.interpolate(dt, 0);

    //    if(change)
    if (change)
    {
        //        renderer.waitIdle();
        //        for(int i = 0; i < 1000; ++i)
        for (auto& v : pointCloud.pointCloud)
        {
            //            Saiga::VertexNC v;
            v.position = make_vec4(linearRand(vec3(-3), vec3(3)), 1);
            v.color    = make_vec4(linearRand(vec3(0), vec3(1)), 1);
            //            pointCloud.mesh.points.push_back(v);
        }
        change        = false;
        uploadChanges = true;
    }

    // camera.setInput(!ImGui::GetIO().WantCaptureKeyboard && !ImGui::GetIO().WantCaptureMouse);
}

void VulkanExample::transfer(vk::CommandBuffer cmd)
{
    assetRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);
    lineAssetRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);
    pointCloudRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);
    texturedAssetRenderer.updateUniformBuffers(cmd, camera.view, camera.proj);


    // upload everything every frame
    if (uploadChanges)
    {
        pointCloud.updateBuffer(cmd, 0, pointCloud.capacity);

        uploadChanges = false;
    }
}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (displayModels)
    {
        if (assetRenderer.bind(cmd))
        {
            assetRenderer.pushModel(cmd, identityMat4());
            plane.render(cmd);
        }

        if (lineAssetRenderer.bind(cmd))
        {
            lineAssetRenderer.pushModel(cmd, translate(vec3(-5, 1.5f, 0)));
            teapot.render(cmd);

            auto gridMatrix = rotate(0.5f * pi<float>(), vec3(1, 0, 0));
            gridMatrix      = translate(gridMatrix, vec3(0, -10, 0));
            lineAssetRenderer.pushModel(cmd, gridMatrix);
            grid.render(cmd);
        }


        if (pointCloudRenderer.bind(cmd))
        {
            pointCloudRenderer.pushModel(cmd, translate(vec3(10, 2.5f, 0)));
            pointCloud.render(cmd, 0, pointCloud.capacity);
        }

        if (texturedAssetRenderer.bind(cmd))
        {
            texturedAssetRenderer.pushModel(cmd, identityMat4());
            texturedAssetRenderer.bindTexture(cmd, box.descriptor);
            box.render(cmd);
        }
    }



    if (textureDisplay.bind(cmd))
    {
        textureDisplay.renderTexture(cmd, textureDes, vec2(10, 10), vec2(100, 50));
    }
}

void VulkanExample::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");
    ImGui::Checkbox("Render models", &displayModels);



    if (ImGui::Button("change point cloud"))
    {
        change = true;
    }


    if (ImGui::Button("reload shader"))
    {
        //        texturedAssetRenderer.shaderPipeline.reload();
        texturedAssetRenderer.reload();
        assetRenderer.reload();
    }


    ImGui::End();
    //    return;

    parentWindow.renderImGui();
    //    ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
    //    ImGui::ShowTestWindow();
}


void VulkanExample::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            parentWindow.close();
            break;
        default:
            break;
    }
}

void VulkanExample::keyReleased(SDL_Keysym key) {}
