/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "openMeshSample.h"

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/assets/objAssetLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/imgui/imgui.h"
#include "saiga/geometry/half_edge_mesh.h"

#include "saiga/geometry/openMeshWrapper.h"

#include "OpenMesh/Tools/Decimater/DecimaterT.hh"
#include "OpenMesh/Tools/Decimater/ModQuadricT.hh"
#include "OpenMesh/Tools/Decimater/ModProgMeshT.hh"
#include "OpenMesh/Tools/Decimater/ModHausdorffT.hh"
#include "OpenMesh/Tools/Decimater/ModAspectRatioT.hh"
#include "OpenMesh/Tools/Decimater/ModEdgeLengthT.hh"
#include "OpenMesh/Tools/Decimater/ModIndependentSetsT.hh"
#include "OpenMesh/Tools/Decimater/ModNormalDeviationT.hh"
#include "OpenMesh/Tools/Decimater/ModNormalFlippingT.hh"
#include "OpenMesh/Tools/Decimater/ModRoundnessT.hh"



using namespace OpenMesh;

SimpleWindow::SimpleWindow(OpenGLWindow *window): Program(window)
{
    //this simplifies shader debugging
    ShaderLoader::instance()->addLineDirectives = true;

    //create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f,aspect,0.1f,50.0f);
    camera.setView(vec3(0,5,10),vec3(0,0,0),vec3(0,1,0));
    camera.enableInput();
    //How fast the camera moves
    camera.movementSpeed = 10;
    camera.movementSpeedFast = 20;

    //Set the camera from which view the scene is rendered
    window->setCamera(&camera);


    //add this object to the keylistener, so keyPressed and keyReleased will be called
    SDL_EventHandler::addKeyListener(this);

    //This simple AssetLoader can create assets from meshes and generate some generic debug assets
    ObjAssetLoader assetLoader;


    assetLoader.loadMeshNC("objs/bunny.obj",baseMesh);
    reducedMesh = baseMesh;

    //    auto bunnyAsset = assetLoader.loadBasicAsset("objs/bunny.obj");
    auto bunnyAsset = assetLoader.assetFromMesh(baseMesh);
    cube1.asset = bunnyAsset;
    cube1.translateGlobal(vec3(0,1,0));
    cube1.calculateModel();

    auto bunnyAsset2 = assetLoader.assetFromMesh(reducedMesh);
    cube2.asset = bunnyAsset2;
    cube2.translateGlobal(vec3(-0,1,0));
    cube2.calculateModel();


#if 0
    auto sphereMesh = TriangleMeshGenerator::createTesselatedPlane(5,5);
    SAIGA_ASSERT(sphereMesh->isValid());
    HalfEdgeMesh<VertexNT,GLuint> hem(*sphereMesh);
    SAIGA_ASSERT(hem.isValid());

    for(int i = 0; i< 100;++i)
    {
        cout << "flip "<< i << endl;
        //        hem.flipEdge(i);
        SAIGA_ASSERT(hem.isValid());
    }
    //    hem.halfEdgeCollapse(0);
    //    hem.halfEdgeCollapse(5);
    //    hem.halfEdgeCollapse(2);
    //    hem.halfEdgeCollapse(10);
    //    hem.halfEdgeCollapse(15);

    //    for(int i = 0; i < 100; ++i)
    //        hem.halfEdgeCollapse(i);
    //    hem.removeFace(0);
    SAIGA_ASSERT(hem.isValid());
    auto ifs = hem.toIFS();
    SAIGA_ASSERT(ifs.isValid());

    auto sphereAsset = assetLoader.assetFromMesh(ifs,Colors::green);

    parentWindow->getRenderer()->wireframe = true;

#endif


#if 0
    auto& ifs = bunnyAsset->mesh;

    using MyMesh = OpenMesh::TriMesh_ArrayKernelT<>;
    MyMesh test;

    std::vector<MyMesh::VertexHandle> handles(ifs.vertices.size());
    for(int i = 0; i < ifs.vertices.size();++i)
    {
        vec3 p = ifs.vertices[i].position;
        handles[i] = test.add_vertex(MyMesh::Point(p.x,p.y,p.z));
    }

    for(int i = 0; i < ifs.faces.size();++i)
    {
        auto f = ifs.faces[i];

        std::vector<MyMesh::VertexHandle> face_vhandles;
        face_vhandles.push_back(handles[f.v1]);
        face_vhandles.push_back(handles[f.v2]);
        face_vhandles.push_back(handles[f.v3]);
        test.add_face(face_vhandles);
    }


    typedef Decimater::DecimaterT<MyMesh>          Decimater;
    //    typedef Decimater::ModQuadricT<MyMesh> MyModQuadric;
    using HModQuadric = OpenMesh::Decimater::ModQuadricT<MyMesh>::Handle;
    //    using HModQuadric = OpenMesh::Decimater::ModProgMeshT<MyMesh>::Handle;
    //    using HModQuadric = OpenMesh::Decimater::ModHausdorffT<MyMesh>::Handle;
    //    Mesh        mesh;             // a mesh object
#if 1
    Decimater   decimater(test);  // a decimater object, connected to a mesh
    HModQuadric hModQuadric;      // use a quadric module
    decimater.add(hModQuadric); // register module at the decimater
    std::cout << decimater.module(hModQuadric).name() << std::endl; // module access

    OpenMesh::Decimater::ModHausdorffT<MyMesh>::Handle dec;
    //    dec.initialize();
    decimater.add(dec);
    /*
     * since we need exactly one priority module (non-binary)
     * we have to call set_binary(false) for our priority module
     * in the case of HModQuadric, unset_max_err() calls set_binary(false) internally
     */
    decimater.module(hModQuadric).unset_max_err();


    //    cout << "tolerance: " << decimater.module(hModQuadric).tolerance() << endl;


    decimater.initialize();
    decimater.decimate(20000);
    // after decimation: remove decimated elements from the mesh
    test.garbage_collection();

    //     dec.decimate();
    //      test.garbage_collection();
#endif


    // write mesh to output.obj
    try
    {
        if ( !OpenMesh::IO::write_mesh(test, "output.off") )
        {
            std::cerr << "Cannot write mesh to file 'output.off'" << std::endl;

        }
    }
    catch( std::exception& x )
    {
        std::cerr << x.what() << std::endl;

    }

    exit(0);
#endif

    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20,20),1.0f,Colors::lightgray,Colors::gray);

    //create one directional light
    sun = window->getRenderer()->lighting.createDirectionalLight();
    sun->setDirection(vec3(-1,-3,-2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(1.0);
    sun->setAmbientIntensity(0.3f);
    sun->createShadowMap(2048,2048);
    sun->enableShadows();

    cout<<"Program Initialized!"<<endl;
}

SimpleWindow::~SimpleWindow()
{
    //We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}

void SimpleWindow::reduce()
{
    using MyMesh = OpenMesh::TriMesh_ArrayKernelT<>;
    MyMesh test;
    triangleMeshToOpenMesh(baseMesh,test);


    // =========================================================================================================


    typedef Decimater::DecimaterT<MyMesh>          Decimater;
    Decimater   decimater(test);

    using HModQuadric = OpenMesh::Decimater::ModQuadricT<MyMesh>::Handle;
    using HModAspect = OpenMesh::Decimater::ModAspectRatioT<MyMesh>::Handle;


    HModQuadric hModQuadric,hModQuadric2;
    HModAspect hModAspect;
//    HModAspect hModAspect;

    decimater.add(hModQuadric2);
    decimater.module(hModQuadric2).unset_max_err();

    if(useQuadric)
    {
        decimater.add(hModQuadric);
        decimater.module(hModQuadric).set_max_err(quadricMaxError);
    }

    if(useAspectRatio)
    {
        decimater.add(hModAspect);
        decimater.module(hModAspect).set_aspect_ratio(ratio);
//        decimater.module(hModAspect).set_error_tolerance_factor(errorTolerance);
    }


    OpenMesh::Decimater::ModHausdorffT<MyMesh>::Handle haus;
    if(useHausdorf)
    {
        decimater.add(haus);
        decimater.module(haus).set_tolerance (hausError);
    }

    OpenMesh::Decimater::ModNormalDeviationT<MyMesh>::Handle nd;
    if(useNormalDev)
    {
        decimater.add(nd);
        decimater.module(nd).set_normal_deviation(normalDev);
    }

    OpenMesh::Decimater::ModNormalFlippingT<MyMesh>::Handle nf;
    if(useNormalFlip)
    {
        decimater.add(nf);
        decimater.module(nf).set_max_normal_deviation(maxNormalDev);
    }

    OpenMesh::Decimater::ModRoundnessT<MyMesh>::Handle r;
    if(useRoundness)
    {
        decimater.add(r);
        decimater.module(r).set_min_roundness(minRoundness);
    }



    decimater.initialize();
    decimater.decimate();


    test.garbage_collection();


    //==============================================================================================

    if(writeToFile)
    {
        saveOpenMesh(test,"output.off");
    }

    //==============================================================================================


    openMeshToTriangleMesh(test,reducedMesh);


    reducedMesh.computePerVertexNormal();



    AssetLoader assetLoader;
    auto bunnyAsset2 = assetLoader.assetFromMesh(reducedMesh);
    cube2.asset = bunnyAsset2;
}

void SimpleWindow::update(float dt){
    //Update the camera position
    camera.update(dt);
    sun->fitShadowToCamera(&camera);
}

void SimpleWindow::interpolate(float dt, float interpolation) {
    //Update the camera rotation. This could also be done in 'update' but
    //doing it in the interpolate step will reduce latency
    camera.interpolate(dt,interpolation);
}

void SimpleWindow::render(Camera *cam)
{
    //Render all objects from the viewpoint of 'cam'
    //    groundPlane.render(cam);
    if(showReduced)
        cube2.render(cam);
    else
        cube1.render(cam);
}

void SimpleWindow::renderDepth(Camera *cam)
{
    //Render the depth of all objects from the viewpoint of 'cam'
    //This will be called automatically for shadow casting light sources to create shadow maps
    //    groundPlane.renderDepth(cam);
    if(showReduced)
        cube2.renderDepth(cam);
    else
        cube1.renderDepth(cam);
}

void SimpleWindow::renderOverlay(Camera *cam)
{
    //The skybox is rendered after lighting and before post processing
    skybox.render(cam);




    if(wireframe)
    {
        glEnable(GL_POLYGON_OFFSET_LINE);
        glPolygonOffset(-10,-10);
        if(showReduced)
            cube2.renderWireframe(cam);
        else
            cube1.renderWireframe(cam);
        glDisable(GL_POLYGON_OFFSET_LINE);
        assert_no_glerror();

    }

}

void SimpleWindow::renderFinal(Camera *cam)
{
    //The final render path (after post processing).
    //Usually the GUI is rendered here.

    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400,200), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

        if(ImGui::CollapsingHeader("Decimation"))
        {

        ImGui::Checkbox("useQuadric",&useQuadric);
        ImGui::SameLine();
        ImGui::InputFloat("quadricMaxError",&quadricMaxError);

        ImGui::Checkbox("useAspectRatio",&useAspectRatio);
        ImGui::SameLine();
        ImGui::InputFloat("ratio",&ratio);
        ImGui::InputFloat("errorTolerance",&errorTolerance);



        ImGui::Checkbox("useHausdorf",&useHausdorf);
        ImGui::SameLine();
        ImGui::InputFloat("hausError",&hausError);

        ImGui::Checkbox("useNormalDev",&useNormalDev);
        ImGui::SameLine();
        ImGui::InputFloat("normalDev",&normalDev);



        ImGui::Checkbox("useNormalFlip",&useNormalFlip);
        ImGui::SameLine();
        ImGui::InputFloat("maxNormalDev",&maxNormalDev);


        ImGui::Checkbox("useRoundness",&useRoundness);
        ImGui::SameLine();
        ImGui::InputFloat("minRoundness",&minRoundness);
        }



        ImGui::Separator();



        ImGui::Checkbox("showReduced",&showReduced);
        ImGui::Checkbox("wireframe",&wireframe);
        ImGui::Checkbox("writeToFile",&writeToFile);


        if(ImGui::Button("bla2"))
        {
            OpenTriangleMesh mesh;
            loadOpenMesh(mesh,"output2.off");

            openMeshToTriangleMesh(mesh,baseMesh);

            baseMesh.computePerVertexNormal();

            AssetLoader assetLoader;
            auto bunnyAsset = assetLoader.assetFromMesh(baseMesh);
            cube1.asset = bunnyAsset;
        }

        if(ImGui::Button("bla"))
        {
            reduce();
        }


        ImGui::Separator();
        ImGui::Text("Base Mesh: V %d F %d ",(int)baseMesh.vertices.size(),(int)baseMesh.faces.size());
        ImGui::Text("Reduced Mesh: V %d F %d ",(int)reducedMesh.vertices.size(),(int)reducedMesh.faces.size());


        ImGui::End();
    }
}


void SimpleWindow::keyPressed(SDL_Keysym key)
{
    switch(key.scancode){
    case SDL_SCANCODE_ESCAPE:
        parentWindow->close();
        break;
    case SDL_SCANCODE_BACKSPACE:
        parentWindow->getRenderer()->printTimings();
        break;
    case SDL_SCANCODE_R:
        ShaderLoader::instance()->reload();
        break;
    case SDL_SCANCODE_F11:
        parentWindow->screenshotRenderDepth("depth.png");
        break;
    case SDL_SCANCODE_F12:
        parentWindow->screenshot("screenshot.png");
        break;
    default:
        break;
    }
}

void SimpleWindow::keyReleased(SDL_Keysym key)
{
}



