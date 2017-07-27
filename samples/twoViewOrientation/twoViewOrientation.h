/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/renderer.h"
#include "saiga/world/proceduralSkybox.h"

#include "saiga/assets/all.h"
#include "saiga/assets/objAssetLoader.h"

#include "saiga/sdl/sdl_eventhandler.h"
#include "saiga/sdl/sdl_camera.h"
#include "saiga/sdl/sdl_window.h"

#include "saiga/rendering/lighting/point_light.h"

#include "saiga/text/all.h"
#include "saiga/rendering/overlay/deferredDebugOverlay.h"
#include "saiga/rendering/overlay/textDebugOverlay.h"
#include "saiga/imgui/imgui_impl_sdl_gl3.h"
#include "saiga/rendering/lighting/directional_light.h"
#include "saiga/geometry/circle.h"

using namespace Saiga;

class AdvancedWindow : public Program, public SDL_KeyListener, public SDL_MouseListener
{
private:

    vec3 c1,c2;

    Circle circle1;
    Circle circle2;
//    Plane circlePlane1;
//    Plane circlePlane2;
//    vec3 circleCenter1, circleCenter2;

    vec3 wp;
    vec3 e1,e2;
    vec3 centerNormal;

public:

    SDLCamera<PerspectiveCamera> camera;

    SimpleAssetObject cube1, cube2;
    SimpleAssetObject groundPlane;
    SimpleAssetObject disc;

    ProceduralSkybox skybox;

    DeferredDebugOverlay ddo;
    TextDebugOverlay tdo;
    ImGui_SDL_Renderer imgui;
    TextureAtlas textAtlas;

    std::vector<std::shared_ptr<PointLight>> lights;
    ObjAssetLoader assetLoader;

    std::shared_ptr<TriangleMesh<VertexNT,GLuint>> circle;
    std::shared_ptr<TriangleMesh<VertexNT,GLuint>> transformedCircle;

    vec3 gradient = vec3(1,0,1);
     vec3 gradientPos;
    vec3 surfaceNormal = vec3(0,1,0);
    vec3 cameraPos = vec3(0,5,0);
    glm::mat3 vbase;

      const float maxAngle = 60;
    float rotationSpeed = 0.1;
    bool showddo = false;
    bool showimguidemo = false;
    bool lightDebug = false;
    bool pointLightShadows = false;
    bool clampNormals = false;
    std::shared_ptr<DirectionalLight> sun;

    AdvancedWindow(OpenGLWindow* window);
    ~AdvancedWindow();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera *cam) override;
    void renderDepth(Camera *cam) override;
    void renderOverlay(Camera *cam) override;
    void renderFinal(Camera *cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;

    void mouseMoved(int x, int y) override;
    void mousePressed(int key, int x, int y) {}
    void mouseReleased(int key, int x, int y) {}

    void projectGeometryToSurface();
    float computeSecondCameraOrientation(vec3 n);
    void displayAngles(std::vector<float> angles);
    vec3 clampNormal(vec3 n);

    vec3 clampNormal2(vec3 n);
};


