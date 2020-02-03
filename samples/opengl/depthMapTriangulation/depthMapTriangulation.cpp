/**
 * Copyright (c) 2020 Simon Mederer
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "depthMapTriangulation.h"

#include "saiga/core/geometry/half_edge_mesh.h"
#include "saiga/core/geometry/openMeshWrapper.h"
#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/vision/util/DepthmapPreprocessor.h"

#include "OpenMesh/Tools/Decimater/DecimaterT.hh"
#include "OpenMesh/Tools/Decimater/ModAspectRatioT.hh"
#include "OpenMesh/Tools/Decimater/ModEdgeLengthT.hh"
#include "OpenMesh/Tools/Decimater/ModHausdorffT.hh"
#include "OpenMesh/Tools/Decimater/ModIndependentSetsT.hh"
#include "OpenMesh/Tools/Decimater/ModNormalDeviationT.hh"
#include "OpenMesh/Tools/Decimater/ModNormalFlippingT.hh"
#include "OpenMesh/Tools/Decimater/ModProgMeshT.hh"
#include "OpenMesh/Tools/Decimater/ModQuadricT.hh"
#include "OpenMesh/Tools/Decimater/ModRoundnessT.hh"

Sample::Sample() : StandaloneWindow("config.ini")
{
    // create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);

    // slow movement for fine adjustments
    camera.movementSpeedFast = 1;
    camera.mouseTurnLocal    = true;

    // Set the camera from which view the scene is rendered
    window->setCamera(&camera);

    // set an all-white background
    renderer->params.clearColor = vec4(1, 1, 1, 1);

    // set the camera parameters of all settings to the values of the used sample image
    StereoCamera4Base<float> cameraParameters = StereoCamera4Base<float>(
        5.3887405952849110e+02, 5.3937051275591125e+02, 3.2233507920081263e+02, 2.3691517848391885e+02, 40.0f);
    ips.cameraParameters = cameraParameters;
	sts.cameraParameters = cameraParameters;
	rqts.cameraParameters = cameraParameters;

    // The necessary calls to get a naive triangulated mesh
    load_depth_image();
    scale_down_depth_image();
    preprocess_occlusion_edges();
    blur_depth_image();
    triangulate_naive();


    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    ObjAssetLoader assetLoader;

    meshObject.asset = assetLoader.assetFromMesh(depthmesh);
    meshObject.translateGlobal(vec3(0, 1, 0));
    meshObject.calculateModel();

    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample() {}

void Sample::load_depth_image()
{
    loaded_depth_image = TemplatedImage<float>(depth_image_input);

    rqts.image_height = loaded_depth_image.height;
    rqts.image_width  = loaded_depth_image.width;
	
    // set the camera parameters of all settings to the values of the used sample image
    cameraParameters = StereoCamera4Base<float>(5.3887405952849110e+02, 5.3937051275591125e+02, 3.2233507920081263e+02,
                                                2.3691517848391885e+02, 40.0f);
	ips.cameraParameters = cameraParameters;
	sts.cameraParameters = cameraParameters;
	rqts.cameraParameters = cameraParameters;
}

void Sample::scale_down_depth_image()
{
    DMPPParameters dmppp = DMPPParameters();

    Intrinsics4 intrinsics =
        Intrinsics4(cameraParameters.fx, cameraParameters.fy, cameraParameters.cx, cameraParameters.cy);

    DMPP dmpp = DMPP(intrinsics, dmppp);

    if (loaded_depth_image.height % 2 != 0 || loaded_depth_image.width % 2 != 0)
    {
        std::cout << "can't scale down image with odd dimension.\n";
        return;
    }

    TemplatedImage<float> result = TemplatedImage<float>(loaded_depth_image.height / 2, loaded_depth_image.width / 2);

    // scale down the image
    dmpp.scaleDown2median(loaded_depth_image, result);

    // also scale down camera parameters
    cameraParameters.scale(0.5f);
	ips.cameraParameters = cameraParameters;
	sts.cameraParameters = cameraParameters;
	rqts.cameraParameters = cameraParameters;

    rqts.image_height /= 2;
    rqts.image_width /= 2;

    loaded_depth_image = result;
}

void Sample::preprocess_occlusion_edges()
{
    ImageProcessor ip(ips);

    ip.remove_occlusion_edges(loaded_depth_image);
}

void Sample::blur_depth_image()
{
    TemplatedImage<float> result(loaded_depth_image.height, loaded_depth_image.width);

    ImageProcessor ip(ips);

    ip.filter_gaussian(loaded_depth_image, result);

    loaded_depth_image = result;
}

void Sample::triangulate_naive()
{
    OpenTriangleMesh m;

    SimpleTriangulator::Settings settings;
    settings.broken_values    = 0.0f;
    settings.cameraParameters = cameraParameters;
    SimpleTriangulator t      = SimpleTriangulator(settings);
    t.triangulate_image(loaded_depth_image, m);

    openMeshToTriangleMesh(m, depthmesh);
    copyVertexColor(m, depthmesh);
    depthmesh.computePerVertexNormal();


    AssetLoader assetLoader;
    meshObject.asset = assetLoader.assetFromMesh(depthmesh);
}

void Sample::triangulate_RQT()
{
    OpenTriangleMesh m;

    RQT_Triangulator t(rqts);

    t.triangulate_image(loaded_depth_image, m);

    openMeshToTriangleMesh(m, depthmesh);
    copyVertexColor(m, depthmesh);
    depthmesh.computePerVertexNormal();

    AssetLoader assetLoader;
    meshObject.asset = assetLoader.assetFromMesh(depthmesh);
}

void Sample::reduce_quadric()
{
    OpenTriangleMesh mesh;

    triangleMeshToOpenMesh(depthmesh, mesh);
    copyVertexColor(depthmesh, mesh);

    QuadricDecimater qd(qds);

    qd.decimate_quadric(mesh);

    openMeshToTriangleMesh(mesh, depthmesh);
    copyVertexColor(mesh, depthmesh);
    depthmesh.computePerVertexNormal();

    AssetLoader assetLoader;
    meshObject.asset = assetLoader.assetFromMesh(depthmesh);
}


void Sample::update(float dt)
{
    // Update the camera position
    if (!ImGui::captureKeyboard()) camera.update(dt);
}

void Sample::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    if (!ImGui::captureMouse()) camera.interpolate(dt, interpolation);
}


void Sample::renderOverlay(Camera* cam)
{
    meshObject.renderForward(cam);

    if (wireframe)
    {
        glEnable(GL_POLYGON_OFFSET_LINE);
        glPolygonOffset(-10, -10);
        meshObject.renderWireframe(cam);
        glDisable(GL_POLYGON_OFFSET_LINE);
        assert_no_glerror();
    }
}

void Sample::renderFinal(Camera* cam)
{
    // The final render path (after post processing).
    // Usually the GUI is rendered here.

    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

        ImGui::InputText("current image", depth_image_input, 100);

        if (ImGui::CollapsingHeader("Preprocess Options"))
        {
            ImGui::InputFloat("hyst min", &ips.hyst_min);
            ImGui::InputFloat("hyst max", &ips.hyst_max);
            ImGui::InputFloat("gauss deviation", &ips.gauss_deviation);
            ImGui::InputInt("gauss radius", &ips.gauss_radius);
        }
        if (ImGui::CollapsingHeader("RQT Options"))
        {
            ImGui::InputFloat("RQT threshold", &rqts.RQT_error_threshold, 0.0f, 0.0f, "%.10f");
        }
        if (ImGui::CollapsingHeader("Reduction Options"))
        {
            ImGui::InputInt("max_decimations", &qds.max_decimations);
            ImGui::InputFloat("quadricMaxError", &qds.quadricMaxError, 0.0f, 0.0f, "%.10f");
            ImGui::Checkbox("check_self_intersections", &qds.check_self_intersections);
            ImGui::Checkbox("check_folding_triangles", &qds.check_folding_triangles);
            ImGui::Checkbox("only_collapse_roughly_parallel_borders", &qds.only_collapse_roughly_parallel_borders);
            ImGui::Separator();
            ImGui::Checkbox("check_interior_angles", &qds.check_interior_angles);
            ImGui::InputFloat("minimal_interior_angle_rad", &qds.minimal_interior_angle_rad, 0.0f, 0.0f, "%.6f");
        }

        ImGui::Separator();
        if (ImGui::Button("1 load depth image"))
        {
            load_depth_image();
            scale_down_depth_image();
        }
        ImGui::Separator();
        if (ImGui::Button("2 preprocess depth image"))
        {
            preprocess_occlusion_edges();
            blur_depth_image();
        }
        ImGui::Separator();
        if (ImGui::Button("3 triangulate naive"))
        {
            triangulate_naive();
        }
        if (ImGui::Button("3 triangulate RQT"))
        {
            triangulate_RQT();
        }
        ImGui::Separator();
        if (ImGui::Button("4 reduce_quadric"))
        {
            reduce_quadric();
        }

        ImGui::Separator();
        ImGui::Checkbox("wireframe", &wireframe);

        ImGui::Separator();
        ImGui::Text("Primitives: V %d F %d ", (int)depthmesh.vertices.size(), (int)depthmesh.faces.size());

        ImGui::End();
    }
}
void Sample::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            window->close();
            break;
        default:
            break;
    }
}

void Sample::keyReleased(SDL_Keysym key) {}


int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
