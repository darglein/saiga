#pragma once

class Renderer {
    public:
    bool wireframe = false;
    float wireframeLineSize = 1;
    Renderer(){}
    virtual void render() = 0;
    virtual void render_intern() = 0;
};


