#pragma once

#include <saiga/opengl/opengl.h>
#include <saiga/camera/camera.h>
#include <saiga/util/mouse.h>

#include <saiga/util/glm.h>


#include <array>



template<typename camera_t>
class Controllable_Camera : public camera_t{
public:
    float movementSpeed = 5;
    float movementSpeedFast = 10;

    float rotationSpeed = 0.2f;

    enum Key{
        Forward = 0,
        Backward = 1,
        Left = 2,
        Right = 3,
        Fast = 4,
        Up = 5,
    };

    const int Dragging = 0;

    std::vector<int> keyboardmap;
    std::vector<int> mousemap;
    glm::ivec2 lastMousePos;

    std::array<bool,6> keyPressed {};

    bool input = true;


    Controllable_Camera(){}
    virtual ~Controllable_Camera(){}

    void update(float delta);
    void interpolate(float dt, float interpolation);

    void mouseRotate(float dx, float dy);

    void enableInput() { input = true; }
    void disableInput() { input = false; }

};

template<class camera_t>
void Controllable_Camera<camera_t>::mouseRotate(float dx, float dy){
    this->turn(dx*rotationSpeed,dy*rotationSpeed);
    this->calculateModel();
    this->updateFromModel();
}

template<class camera_t>
void Controllable_Camera<camera_t>::update(float delta){
    if(!input)
        return;
    int FORWARD = keyboard.getMappedKeyState(Forward,keyboardmap) - keyboard.getMappedKeyState(Backward,keyboardmap);
    int RIGHT = keyboard.getMappedKeyState(Right,keyboardmap) - keyboard.getMappedKeyState(Left,keyboardmap);

    float speed;
    if (keyboard.getMappedKeyState(Fast,keyboardmap)){
        speed = movementSpeedFast;
    } else {
        speed = movementSpeed;
    }

    vec3 trans = delta*speed*FORWARD*vec3(0,0,-1) + delta*speed*RIGHT*vec3(1,0,0);
    vec3 transg =  vec3(0,1,0) * (delta*speed*keyboard.getMappedKeyState(Up,keyboardmap));
    this->translateLocal(trans);
    this->translateGlobal(transg);
    this->calculateModel();
    this->updateFromModel();
}


template<class camera_t>
void Controllable_Camera<camera_t>::interpolate(float dt, float interpolation){
    if(!input)
        return;

    //only do mouse handling here
    glm::ivec2 mousedelta = lastMousePos - mouse.getPosition();
    lastMousePos = mouse.getPosition();

    if(mouse.getMappedKeyState(Dragging,mousemap)){
        this->mouseRotate(mousedelta.x,mousedelta.y);
    }

}
