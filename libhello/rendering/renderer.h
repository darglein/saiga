#pragma once

class Camera;

class Renderer {
    public:
    bool wireframe = false;
    float wireframeLineSize = 1;
    Renderer(){}
    virtual void render(Camera *cam, float interpolation = 0.f) = 0;
    virtual void renderDepth(Camera *cam) = 0;
    virtual void render_intern(float interpolation = 0.f) = 0;
};


