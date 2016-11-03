#pragma once

#include "saiga/camera/controllable_camera.h"
#include <GLFW/glfw3.h>

template<typename camera_t>
class Glfw_Camera : public Controllable_Camera<camera_t>{
public:
    Glfw_Camera(){
        this->keyboardmap = {
            GLFW_KEY_W,
            GLFW_KEY_S,
            GLFW_KEY_A,
            GLFW_KEY_D,
//            GLFW_KEY_UP,
//            GLFW_KEY_DOWN,
//            GLFW_KEY_LEFT,
//            GLFW_KEY_RIGHT,
            GLFW_KEY_LEFT_SHIFT,
            GLFW_KEY_SPACE
        };
        this->mousemap = {
            GLFW_MOUSE_BUTTON_3
        };
    }
};


