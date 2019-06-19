/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES
#endif

#include "openMeshSample.h"

#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/core/geometry/half_edge_mesh.h"
#include "saiga/core/geometry/openMeshWrapper.h"
#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"

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



using namespace OpenMesh;

Sample::Sample(OpenGLWindow& window, Renderer& renderer) : Updating(window), DeferredRenderingInterface(renderer)
{
    // create a perspective camera
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.enableInput();

    // Set the camera from which view the scene is rendered
    window.setCamera(&camera);


    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    ObjAssetLoader assetLoader;


    assetLoader.loadMeshNC("box.obj", baseMesh);
    reducedMesh = baseMesh;

    //    auto bunnyAsset = assetLoader.loadBasicAsset("objs/bunny.obj");
    auto bunnyAsset = assetLoader.assetFromMesh(baseMesh);
    cube1.asset     = bunnyAsset;
    cube1.translateGlobal(vec3(0, 1, 0));
    cube1.calculateModel();

    auto bunnyAsset2 = assetLoader.assetFromMesh(reducedMesh);
    cube2.asset      = bunnyAsset2;
    cube2.translateGlobal(vec3(-0, 1, 0));
    cube2.calculateModel();

    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20, 20), 1.0f, Colors::lightgray, Colors::gray);

    // create one directional light
    Deferred_Renderer& r = static_cast<Deferred_Renderer&>(parentRenderer);
    sun                  = r.lighting.createDirectionalLight();
    sun->setDirection(vec3(-1, -3, -2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(1.0);
    sun->setAmbientIntensity(0.3f);
    sun->createShadowMap(2048, 2048);
    sun->enableShadows();

    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample()
{
    // We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}


template <class MeshT>
class ModNone : public OpenMesh::Decimater::ModBaseT<MeshT>
{
   public:
    // Defines the types Self, Handle, Base, Mesh, and CollapseInfo
    // and the memberfunction name()
    DECIMATING_MODULE(ModNone, MeshT, None);

    ModNone(MeshT& _mesh) : Base(_mesh, false) { Base::set_binary(false); }


    virtual float collapse_priority(const CollapseInfo& _ci) { return 0; }

    virtual void initialize(void) { Base::set_binary(false); }
};


void Sample::reduce()
{
#ifdef OM_DEBUG
    std::cerr << "Warning OpenMesh debug is ON" << std::endl;
#endif


    using MyMesh = OpenTriangleMesh;
    MyMesh mesh;

    {
        ScopedTimerPrint tim("convert mesh");

        triangleMeshToOpenMesh(baseMesh, mesh);
        copyVertexColor(baseMesh, mesh);
    }


    mesh.request_face_normals();
    mesh.update_face_normals();

    // =========================================================================================================



    typedef Decimater::DecimaterT<MyMesh> Decimater;
    Decimater decimater(mesh);

    using HModQuadric = OpenMesh::Decimater::ModQuadricT<MyMesh>::Handle;
    using HModAspect  = OpenMesh::Decimater::ModAspectRatioT<MyMesh>::Handle;


    HModQuadric hModQuadric, hModQuadric2;
    HModAspect hModAspect;
    //    HModAspect hModAspect;

    ModNone<MyMesh>::Handle none;


    decimater.add(none);
    //    decimater.module(hModQuadric2).unset_max_err();

    //    decimater.add(hModQuadric2);
    //    decimater.module(hModQuadric2).unset_max_err();

    if (useQuadric)
    {
        decimater.add(hModQuadric);
        decimater.module(hModQuadric).set_max_err(quadricMaxError);
    }

    if (useAspectRatio)
    {
        decimater.add(hModAspect);
        decimater.module(hModAspect).set_aspect_ratio(ratio);
        //        decimater.module(hModAspect).set_error_tolerance_factor(errorTolerance);
    }


    OpenMesh::Decimater::ModHausdorffT<MyMesh>::Handle haus;
    if (useHausdorf)
    {
        decimater.add(haus);
        decimater.module(haus).set_tolerance(hausError);
    }

    OpenMesh::Decimater::ModNormalDeviationT<MyMesh>::Handle nd;
    if (useNormalDev)
    {
        decimater.add(nd);
        decimater.module(nd).set_normal_deviation(normalDev);
    }

    OpenMesh::Decimater::ModNormalFlippingT<MyMesh>::Handle nf;
    if (useNormalFlip)
    {
        decimater.add(nf);
        decimater.module(nf).set_max_normal_deviation(maxNormalDev);
    }

    OpenMesh::Decimater::ModRoundnessT<MyMesh>::Handle r;
    if (useRoundness)
    {
        decimater.add(r);
        decimater.module(r).set_min_roundness(minRoundness);
    }



    decimater.initialize();


    {
        ScopedTimerPrint tim("decimate");
        decimater.decimate();
    }


    mesh.garbage_collection();



    {
        ScopedTimerPrint tim("convert mesh");
        openMeshToTriangleMesh(mesh, reducedMesh);

        copyVertexColor(mesh, reducedMesh);

        reducedMesh.computePerVertexNormal();

        AssetLoader assetLoader;
        auto bunnyAsset2 = assetLoader.assetFromMesh(reducedMesh);
        cube2.asset      = bunnyAsset2;
    }
}

void Sample::update(float dt)
{
    // Update the camera position
    camera.update(dt);
    sun->fitShadowToCamera(&camera);
}

void Sample::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    camera.interpolate(dt, interpolation);
}

void Sample::render(Camera* cam)
{
    // Render all objects from the viewpoint of 'cam'
    //    groundPlane.render(cam);
    if (showReduced)
        cube2.render(cam);
    else
        cube1.render(cam);
}

void Sample::renderDepth(Camera* cam)
{
    // Render the depth of all objects from the viewpoint of 'cam'
    // This will be called automatically for shadow casting light sources to create shadow maps
    //    groundPlane.renderDepth(cam);
    if (showReduced)
        cube2.renderDepth(cam);
    else
        cube1.renderDepth(cam);
}

void Sample::renderOverlay(Camera* cam)
{
    // The skybox is rendered after lighting and before post processing
    skybox.render(cam);



    if (wireframe)
    {
        glEnable(GL_POLYGON_OFFSET_LINE);
        glPolygonOffset(-10, -10);
        if (showReduced)
            cube2.renderWireframe(cam);
        else
            cube1.renderWireframe(cam);
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


        static char fileOff[256] = "output2.off";
        ImGui::InputText("off file", fileOff, 256);

        if (ImGui::Button("Test saiga::halfedge"))
        {
            Saiga::Timer t;

            t.start();
            Saiga::HalfEdgeMesh<VertexNC, GLuint> hem(baseMesh);
            t.stop();

            std::cout << "to halfedge: " << t.getTimeMS() << std::endl;

            {
                ScopedTimerPrint tim("isValid");
                SAIGA_ASSERT(hem.isValid());
            }


            {
                ScopedTimerPrint tim("to ifs");
                TriangleMesh<VertexNC, GLuint> m;
                hem.toIFS(m);
            }
        }


        if (ImGui::Button("Load .off"))
        {
            OpenMesh::TriMesh_ArrayKernelT<> mesh;
            //            OpenTriangleMesh mesh;
            loadOpenMesh(mesh, fileOff);

            openMeshToTriangleMesh(mesh, baseMesh);

            baseMesh.computePerVertexNormal();

            AssetLoader assetLoader;
            auto bunnyAsset = assetLoader.assetFromMesh(baseMesh);
            cube1.asset     = bunnyAsset;
        }

        static char fileObj[256] = "bunny.obj";
        ImGui::InputText("obj file", fileObj, 256);

        if (ImGui::Button("Load .obj"))
        {
            ObjAssetLoader assetLoader;
            assetLoader.loadMeshNC(fileObj, baseMesh);
            auto bunnyAsset = assetLoader.assetFromMesh(baseMesh);
            cube1.asset     = bunnyAsset;
        }

        if (ImGui::CollapsingHeader("Decimation"))
        {
            ImGui::Checkbox("useQuadric", &useQuadric);
            ImGui::SameLine();
            ImGui::InputFloat("quadricMaxError", &quadricMaxError);

            ImGui::Checkbox("useAspectRatio", &useAspectRatio);
            ImGui::SameLine();
            ImGui::InputFloat("ratio", &ratio);
            ImGui::InputFloat("errorTolerance", &errorTolerance);



            ImGui::Checkbox("useHausdorf", &useHausdorf);
            ImGui::SameLine();
            ImGui::InputFloat("hausError", &hausError);

            ImGui::Checkbox("useNormalDev", &useNormalDev);
            ImGui::SameLine();
            ImGui::InputFloat("normalDev", &normalDev);



            ImGui::Checkbox("useNormalFlip", &useNormalFlip);
            ImGui::SameLine();
            ImGui::InputFloat("maxNormalDev", &maxNormalDev);


            ImGui::Checkbox("useRoundness", &useRoundness);
            ImGui::SameLine();
            ImGui::InputFloat("minRoundness", &minRoundness);
        }



        ImGui::Separator();



        ImGui::Checkbox("showReduced", &showReduced);
        ImGui::Checkbox("wireframe", &wireframe);
        ImGui::Checkbox("writeToFile", &writeToFile);



        if (ImGui::Button("Reduce"))
        {
            reduce();
        }


        ImGui::Separator();
        ImGui::Text("Base Mesh: V %d F %d ", (int)baseMesh.vertices.size(), (int)baseMesh.faces.size());
        ImGui::Text("Reduced Mesh: V %d F %d ", (int)reducedMesh.vertices.size(), (int)reducedMesh.faces.size());


        ImGui::End();
    }
}


void Sample::keyPressed(SDL_Keysym key)
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

void Sample::keyReleased(SDL_Keysym key) {}
