/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "twoViewOrientation.h"

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/rendering/lighting/directional_light.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/geometry/intersection.h"


AdvancedWindow::AdvancedWindow(OpenGLWindow *window): Program(window),
    ddo(window->getWidth(),window->getHeight()),tdo(window->getWidth(),window->getHeight())
{
    //this simplifies shader debugging
    ShaderLoader::instance()->addLineDirectives = true;


    //create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f,aspect,0.1f,50.0f);
    camera.setView(cameraPos,vec3(0,0,0),vec3(1,0,0));
    camera.enableInput();
    vbase = glm::mat3(camera.view);
    //How fast the camera moves
    camera.movementSpeed = 10;
    camera.movementSpeedFast = 20;
    camera.recalculateMatrices();

    //Set the camera from which view the scene is rendered
    window->setCamera(&camera);


    //add this object to the keylistener, so keyPressed and keyReleased will be called
    SDL_EventHandler::addKeyListener(this);
    SDL_EventHandler::addMouseListener(this);




    //    auto cubeAsset = assetLoader.loadTexturedAsset("objs/box.obj");


    {
        std::vector<vec3> vs = {
            vec3(0),vec3(1,0,1)
        };
        std::vector<GLuint> is = {0,1};
        cube1.asset = assetLoader.nonTriangleMesh(vs,is,GL_LINES,vec4(1,0,0,1));
        cube1.calculateModel();
    }
    {
        std::vector<vec3> vs = {
            vec3(0),vec3(2,0,1),
            vec3(0),vec3(1,0,1)
        };
        std::vector<GLuint> is = {0,1,2,3};
        cube2.asset = assetLoader.nonTriangleMesh(vs,is,GL_LINES,vec4(0,0,1,1));
        cube2.calculateModel();
    }


    circle = TriangleMeshGenerator::createCircleMesh<VertexNT,GLuint>(100,1);

    transformedCircle = std::make_shared<TriangleMesh<VertexNT,GLuint>>(*circle);

    //    auto sphereAsset = assetLoader.loadBasicAsset("objs/teapot.obj");
    auto sphereAsset = assetLoader.assetFromMesh(transformedCircle);
    disc.asset = sphereAsset;
    //    sphere.translateGlobal(vec3(-2,1,0));
    //    sphere.rotateLocal(vec3(0,1,0),180);
    disc.calculateModel();

    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20,20),1.0f,Colors::lightgray,Colors::gray);
    groundPlane.translateGlobal(vec3(0,-5,0));
    groundPlane.calculateModel();

    //create one directional light
    sun = window->getRenderer()->lighting.createDirectionalLight();
    sun->setDirection(vec3(-1,-3,-2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(0.5);
    sun->setAmbientIntensity(0.1f);
    sun->createShadowMap(2048,2048);
    sun->enableShadows();




    ddo.setDeferredFramebuffer(&window->getRenderer()->gbuffer,window->getRenderer()->ssao.bluredTexture);

    imgui.init(((SDLWindow*)window)->window,"fonts/SourceSansPro-Regular.ttf");

    textAtlas.loadFont("fonts/SourceSansPro-Regular.ttf",40,2,4,true);

    tdo.init(&textAtlas);
    tdo.borderX = 0.01f;
    tdo.borderY = 0.01f;
    tdo.paddingY = 0.000f;
    tdo.textSize = 0.04f;

    tdo.textParameters.setColor(vec4(1),0.1f);
    tdo.textParameters.setGlow(vec4(0,0,0,1),1.0f);

    tdo.createItem("Fps: ");
    tdo.createItem("Ups: ");
    tdo.createItem("Render Time: ");
    tdo.createItem("Update Time: ");


    cout<<"Program Initialized!"<<endl;
}

AdvancedWindow::~AdvancedWindow()
{
    //We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
    imgui.shutdown();
}

void AdvancedWindow::update(float dt){
    //Update the camera position
    camera.update(dt);




    sun->fitShadowToCamera(&camera);
    //    sun->fitNearPlaneToScene(sceneBB);

    int  fps = (int) glm::round(1000.0/parentWindow->fpsTimer.getTimeMS());
    tdo.updateEntry(0,fps);

    int  ups = (int) glm::round(1000.0/parentWindow->upsTimer.getTimeMS());
    tdo.updateEntry(1,ups);

    float renderTime = parentWindow->getRenderer()->getTime(Deferred_Renderer::TOTAL);
    tdo.updateEntry(2,renderTime);

    float updateTime = parentWindow->updateTimer.getTimeMS();
    tdo.updateEntry(3,updateTime);


}

void AdvancedWindow::interpolate(float dt, float interpolation) {
    //Update the camera rotation. This could also be done in 'update' but
    //doing it in the interpolate step will reduce latency
    camera.interpolate(dt,interpolation);
}

void AdvancedWindow::render(Camera *cam)
{
    //Render all objects from the viewpoint of 'cam'
    groundPlane.render(cam);
    //    cube1.render(cam);
    //    cube2.render(cam);
    //    disc.render(cam);
}

void AdvancedWindow::renderDepth(Camera *cam)
{
    //Render the depth of all objects from the viewpoint of 'cam'
    //This will be called automatically for shadow casting light sources to create shadow maps
    groundPlane.renderDepth(cam);
    //    cube1.renderDepth(cam);
    //    cube2.renderDepth(cam);
    //    disc.render(cam);
}

void AdvancedWindow::renderOverlay(Camera *cam)
{
    //The skybox is rendered after lighting and before post processing
    skybox.render(cam);
    disc.renderForward(cam);
    cube1.renderForward(cam);
    cube2.renderForward(cam);
}

static float angleDiff(float base, float a){
    //a = base + diff
    float diff = a - base;
    if(diff > 180)
        diff -= 360;
    if(diff < -180)
        diff += 360;
    return diff;
}

vec3 AdvancedWindow::clampNormal(vec3 n){
    for(int i =0; i < 10; ++i){
        if(glm::degrees(glm::acos(glm::dot(e1,n))) > maxAngle)
            n = circle1.closestPointOnCircle(n);
        if(glm::degrees(glm::acos(glm::dot(e2,n))) > maxAngle)
            n = circle2.closestPointOnCircle(n);
    }
    return n;
}

vec3 AdvancedWindow::clampNormal2(vec3 n){
    for(int i =0; i < 10; ++i){
//        if(glm::degrees(glm::acos(glm::dot(e1,n))) > maxAngle)
//            n = circle1.closestPointOnCircle(n);
        if(glm::degrees(glm::acos(glm::dot(e2,n))) > maxAngle)
            n = circle2.closestPointOnCircle(n);
    }
    return n;
}

void AdvancedWindow::renderFinal(Camera *cam)
{

    //The final render path (after post processing).
    //Usually the GUI is rendered here.

    parentWindow->getRenderer()->bindCamera(&tdo.layout.cam);
    tdo.render();
    if(showddo)
        ddo.render();



    imgui.beginFrame();

    {
        ImGui::SetNextWindowPos(ImVec2(50, 400), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200,400), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

        ImGui::Checkbox("Show Imgui demo", &showimguidemo );
        ImGui::Checkbox("Show deferred debug overlay", &showddo );


        if(ImGui::Button("reset camera")){
            camera.setView(cameraPos,vec3(0,0,0),vec3(1,0,0));
        }

        if(ImGui::Button("reset camera2")){
            camera.setView(cameraPos + vec3(1,0,0),vec3(0,0,0),vec3(1,0,0));
        }


        if(ImGui::Button("reset normal")){
            //            camera.setView(cameraPos + vec3(1,0,0),vec3(0,0,0),vec3(1,0,0));
            vec3 n1 = normalize(cameraPos - vec3(0));
            vec3 n2 = normalize(camera.getPosition() - vec3(0));
            surfaceNormal = normalize(n1 + n2);
        }

        ImGui::Checkbox("clampNormals", &clampNormals );


        //world positions of camera

        c2 = camera.getPosition();
        c1 = cameraPos;
        //world position of triangulated point
        wp = vec3(0);
        e1 = normalize(c1 - wp);
        e2 = normalize(c2 - wp);
        float angleBetweenCameras;
        angleBetweenCameras = glm::degrees(glm::acos(glm::dot(e1,e2)));
        ImGui::Text("angleBetweenCameras: %f\n",angleBetweenCameras);
        centerNormal = normalize(e1 + e2);


        {
            //small circle stuff
            float h = cos(glm::radians(maxAngle));
            float r = sin(glm::radians(maxAngle));
            circle1.r = r;
            circle2.r = r;
            circle1.pos = h * e1;
            circle2.pos = h * e2;
            circle1.normal = e1;
            circle2.normal = e2;
        }
        //        circlePlane2 = Plane(circleCenter2,e2);

        bool valid = true;
        {
            //angle between cameras and center
            float a1 = glm::degrees(glm::acos(glm::dot(e1,centerNormal)));
            float a2 = glm::degrees(glm::acos(glm::dot(e2,centerNormal)));
            ImGui::Text("center angles: %f %f \n",a1,a2);

            if(a1 > maxAngle || a2 > maxAngle)
                valid = false;
        }
        ImGui::Text("valid: %d\n",valid);
        vec3 left,right;
        {
            vec3 u  = normalize(cross(centerNormal,e1));
            float remainingAngle = angleBetweenCameras - maxAngle;
            remainingAngle =  glm::radians(remainingAngle);
            left = glm::angleAxis( remainingAngle, -u) * e1;
            right = glm::angleAxis( remainingAngle, u) * e2;
            left = normalize(left);
            right = normalize(right);
        }
        {
            //test angle between left/right to cameras
            float a1l = glm::degrees(glm::acos(glm::dot(e1,left)));
            float a1r = glm::degrees(glm::acos(glm::dot(e1,right)));
            float a2l = glm::degrees(glm::acos(glm::dot(e2,left)));
            float a2r = glm::degrees(glm::acos(glm::dot(e2,right)));
            ImGui::Text("left right angles: %f %f | %f %f\n",a1l,a1r,a2l,a2r);
        }

        vec3 bottom,top;

        {
            Ray ray;
            Saiga::Intersection::PlanePlane(circle1.getPlane(),circle2.getPlane(),ray);

            Sphere s(wp,1);
            float t1,t2;
            Saiga::Intersection::RaySphere(ray,s,t1,t2);
            bottom = ray.getAlphaPosition(t1);
            top = ray.getAlphaPosition(t2);
            ImGui::Text("bottom: %s %f\n",Saiga::to_string(bottom).c_str(),length(bottom));
            ImGui::Text("top: %s %f\n",Saiga::to_string(top).c_str(),length(top));
        }


        vec3 test1,test2,test3,test4;
        vec3 axis = normalize(gradient - wp);
        axis = normalize(cross(axis,e1));
        //        vec3 axis = normalize(gradient - wp);
        {
            Plane mocp(wp,axis);
            Ray ray;
            Sphere s(wp,1);

            {
                Saiga::Intersection::PlanePlane(circle1.getPlane(),mocp,ray);
                float t1,t2;
                Saiga::Intersection::RaySphere(ray,s,t1,t2);
                test1 = ray.getAlphaPosition(t1);
                test2 = ray.getAlphaPosition(t2);
            }
            {
                Saiga::Intersection::PlanePlane(circle2.getPlane(),mocp,ray);
                float t1,t2;
                Saiga::Intersection::RaySphere(ray,s,t1,t2);
                test3 = ray.getAlphaPosition(t1);
                test4 = ray.getAlphaPosition(t2);
            }
            if(clampNormals){
                test1 = clampNormal2(test1);
                test2 = clampNormal2(test2);
                test3 = clampNormal(test3);
                test4 = clampNormal(test4);
            }
        }

        std::swap(surfaceNormal.z,surfaceNormal.y);
        ImGui::Direction("surfaceNormal2",surfaceNormal);
        std::swap(surfaceNormal.z,surfaceNormal.y);
        surfaceNormal= normalize(surfaceNormal);








        static float angleOriLong = 0;
        if(ImGui::SliderAngle("angleOriLong",&angleOriLong,-180,180)){
            surfaceNormal = cross(axis,cross(e1,e2));
            //            surfaceNormal = cross(axis,e2);
            //            surfaceNormal = e2;
            //            surfaceNormal = centerNormal;
            //            if(dot(surfaceNormal,e1) > 0)
            //                surfaceNormal *= -1;
            surfaceNormal = normalize(surfaceNormal);
            //            cout << surfaceNormal << endl;
            surfaceNormal = glm::angleAxis(angleOriLong,axis) * surfaceNormal;
            //            cout << surfaceNormal << endl;
        }

        surfaceNormal = clampNormal(surfaceNormal);
        projectGeometryToSurface();

        float baseAngle = computeSecondCameraOrientation(centerNormal);
        float a = computeSecondCameraOrientation(surfaceNormal);
        ImGui::Text("base orientation: %f\n",baseAngle);
        ImGui::Text("second camera orientation: %f %f\n",a,angleDiff(baseAngle,a));

        std::vector<float> angles;
        //        angles.push_back(baseAngle);
        //        angles.push_back(a);
        //        angles.push_back(computeSecondCameraOrientation(left));
        //        angles.push_back(computeSecondCameraOrientation(right));
        //        angles.push_back(computeSecondCameraOrientation(bottom));
        //        angles.push_back(computeSecondCameraOrientation(top));
        angles.push_back(computeSecondCameraOrientation(test1));
        angles.push_back(computeSecondCameraOrientation(test2));
        //        angles.push_back(computeSecondCameraOrientation(test3));
        //        angles.push_back(computeSecondCameraOrientation(test4));
        displayAngles(angles);

        groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20,20),1.0f,Colors::lightgray,Colors::gray);
        {
#if 0
            //project to current camera
            vec2 iwp = camera.projectToScreenSpace(vec3(0),parentWindow->getWidth(),parentWindow->getHeight());
            vec2 ior = camera.projectToScreenSpace(gradientPos,parentWindow->getWidth(),parentWindow->getHeight());
            vec2 ori = vec2(ior.x,ior.y) - vec2(iwp.x,iwp.y);
            ori = normalize(ori);
            float angle = glm::degrees(atan2(ori.y,ori.x));

            float minA = 0346436;
            float maxA = -345346;
            vec3 minN;
            vec3 maxN;

            int segments = 100;
            float R = 1./(float)(segments);
            float r = 1;
            for(int s=0;s<segments;s++){
                float x = r * glm::sin((float)s*R*glm::two_pi<float>());
                float y = r * glm::cos((float)s*R*glm::two_pi<float>());

                vec3 n = vec3(x,0,y);
                for(int i =0; i < 10; ++i){
                    n = clampAngle(glm::mat3(camera.view),n);
                    n = clampAngle(glm::mat3(vbase),n);
                }

                Plane plane(vec3(0),n);
                vec3 dir = normalize( vec3(1,0,1) - cameraPos);
                Ray r( dir, cameraPos );
                float t;
                r.intersectPlane(plane,t);
                gradientPos = cameraPos + dir * t;

                vec2 iwp = camera.projectToScreenSpace(vec3(0),parentWindow->getWidth(),parentWindow->getHeight());
                vec2 ior = camera.projectToScreenSpace(gradientPos,parentWindow->getWidth(),parentWindow->getHeight());
                vec2 ori = vec2(ior.x,ior.y) - vec2(iwp.x,iwp.y);
                ori = normalize(ori);
                float angle = glm::degrees(atan2(ori.y,ori.x));
                if(angle < minA){
                    minA = angle;
                    minN = n;
                }

                if(angle > maxA){
                    maxA = angle;
                    maxN = n;
                }
            }

            ImGui::Text("n: %s\n",Saiga::to_string(surfaceNormal).c_str());
            ImGui::Text("minN: %s\n",Saiga::to_string(minN).c_str());
            ImGui::Text("maxN: %s\n",Saiga::to_string(maxN).c_str());
            ImGui::Text("orientation: %f %f %f\n",angle,minA,maxA);

#endif
        }


        ImGui::End();
    }

    if (showimguidemo)
    {
        ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
        ImGui::ShowTestWindow(&showimguidemo);
    }

    imgui.endFrame();

    imgui.checkWindowFocus();
}

float AdvancedWindow::computeSecondCameraOrientation(vec3 n){
    if(length(n) < 0.1)
        return 0;
    Plane plane(vec3(0),n);

    vec3 dir = normalize( gradient - cameraPos);
    Ray r( dir, cameraPos );
    float t;
    r.intersectPlane(plane,t);
    vec3 gradientPos = cameraPos + dir * t;

    vec2 iwp = camera.projectToScreenSpace(vec3(0),parentWindow->getWidth(),parentWindow->getHeight());
    vec2 ior = camera.projectToScreenSpace(gradientPos,parentWindow->getWidth(),parentWindow->getHeight());
    vec2 ori = vec2(ior.x,ior.y) - vec2(iwp.x,iwp.y);
    ori = normalize(ori);
    float angle = glm::degrees(atan2(ori.y,ori.x));
    if(angle<0)
        angle += 360;
    return angle;
}

void AdvancedWindow::displayAngles(std::vector<float> angles){

    std::vector<vec3> vs;
    std::vector<GLuint> is;


    int i= 1;
    for(float a : angles){
        a = glm::radians(a);
        vec2 iwp = camera.projectToScreenSpace(vec3(0),parentWindow->getWidth(),parentWindow->getHeight());
        vec2 ior = iwp + (i*80.0f) * vec2(cos(a),sin(a));

        vec3 wpor = camera.inverseprojectToWorldSpace(ior,1,parentWindow->getWidth(),parentWindow->getHeight());

        vs.push_back(vec3(0));
        vs.push_back(wpor);

        is.push_back(vs.size() - 2);
        is.push_back(vs.size() - 1);
        i++;
    }

    auto cubeAsset = assetLoader.nonTriangleMesh(vs,is,GL_LINES,vec4(0,0,1,1));
    cube2.asset = cubeAsset;

}

void AdvancedWindow::projectGeometryToSurface()
{
    Plane plane(vec3(0),surfaceNormal);
    //project vertex on plane
    for(int i = 0;i < (int)circle->vertices.size(); ++i){
        VertexNT& v1 = circle->vertices[i];
        VertexNT& v2 = transformedCircle->vertices[i];
        vec3 dir = normalize( vec3(v1.position) - cameraPos);
        Ray r( dir, cameraPos );
        float t;
        r.intersectPlane(plane,t);
        v2.position = vec4(cameraPos + dir * t,1);
    }

    disc.asset = assetLoader.assetFromMesh(transformedCircle);;


    {
        std::vector<vec3> vs = {
            vec3(0),gradient
        };
        //project vertex on plane
        for(int i = 0;i < (int)vs.size(); ++i){
            vec3& v1 = vs[i];
            vec3& v2 = vs[i];
            vec3 dir = normalize( v1 - cameraPos);
            Ray r( dir, cameraPos );
            float t;
            r.intersectPlane(plane,t);
            v2 = cameraPos + dir * t;
        }
        gradientPos = vs[1];
        std::vector<GLuint> is = {0,1};
        auto cubeAsset = assetLoader.nonTriangleMesh(vs,is,GL_LINES,vec4(1,0,0,1));
        cube1.asset = cubeAsset;
    }
}

void AdvancedWindow::keyPressed(SDL_Keysym key)
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
    case SDL_SCANCODE_F12:
        parentWindow->screenshot("screenshot.png");
        break;
    default:
        break;
    }
}

void AdvancedWindow::keyReleased(SDL_Keysym key)
{
}

void AdvancedWindow::mouseMoved(int x, int y)
{
    if(imgui.wantsCaptureMouse)
        return;
    vec2 pos(x,y);
    static vec2 lastPos = pos;
    vec2 relMovement = pos - lastPos;
    if(mouse.getKeyState(SDL_BUTTON_LEFT)){
        relMovement *= 0.5;
        camera.mouseRotateAroundPoint(relMovement.x,relMovement.y,vec3(0));
    }
    lastPos = pos;
}




