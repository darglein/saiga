#pragma once

class Camera;

class Renderer {
    public:
    bool wireframe = false;
    float wireframeLineSize = 1;
    Renderer(){}
    virtual void render(Camera *cam) = 0;
    virtual void renderDepth(Camera *cam) = 0;
    virtual void render_intern() = 0;
};


