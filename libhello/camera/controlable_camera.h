#pragma once

#include "libhello/sdl/listener.h"
#include "libhello/util/glm.h"

#include "libhello/window/glfw_eventhandler.h"

#include <SDL2/SDL.h>
#include <iostream>


template<typename camera_t>
class Controlable_Camera : public glfw_KeyListener , public glfw_MouseListener{
public:
    double lastmx=0,lastmy=0;
    float FORWARD=0,RIGHT=0;
    camera_t* cam;
    bool dragging = false;
    float speed = 1;
    Controlable_Camera(camera_t* cam):cam(cam){ glfw_EventHandler::addKeyListener(this);glfw_EventHandler::addMouseListener(this);}
    void update(float delta);


    //glfw events
    bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods);
    bool character_event(GLFWwindow* window, unsigned int codepoint);
    bool cursor_position_event(GLFWwindow* window, double xpos, double ypos);
    bool mouse_button_event(GLFWwindow* window, int button, int action, int mods);
    bool scroll_event(GLFWwindow* window, double xoffset, double yoffset);


};

template<class camera_t>
void Controlable_Camera<camera_t>::update(float delta){
    vec3 trans = delta*speed*FORWARD*vec3(0,0,-1) + delta*speed*RIGHT*vec3(1,0,0);
    cam->translateLocal(trans);
    cam->updateFromModel();
}




template<class camera_t>
bool Controlable_Camera<camera_t>::key_event(GLFWwindow* window, int key, int scancode, int action, int mods){
    switch(key){
    case GLFW_KEY_UP:
        FORWARD = (action!=GLFW_RELEASE  )?1:0;
        return true;
    case GLFW_KEY_DOWN:
        FORWARD = (action!=GLFW_RELEASE)?-1:0;
        return true;
    case GLFW_KEY_RIGHT:
        RIGHT = (action!=GLFW_RELEASE)?1:0;
        return true;
    case GLFW_KEY_LEFT:
        RIGHT = (action!=GLFW_RELEASE)?-1:0;
        return true;
    }
    return false;
}


template<class camera_t>
bool Controlable_Camera<camera_t>::character_event(GLFWwindow* window, unsigned int codepoint){
    return false;
}

template<class camera_t>
bool Controlable_Camera<camera_t>::cursor_position_event(GLFWwindow* window, double xpos, double ypos){

    if(dragging){
        float relx = lastmx-xpos;
        float rely = lastmy-ypos;
        cam->turn(relx,rely);
        cam->updateFromModel();
    }
    lastmx = xpos;
    lastmy = ypos;
    return false;
}

template<class camera_t>
bool Controlable_Camera<camera_t>::mouse_button_event(GLFWwindow* window, int button, int action, int mods){
    if(button==GLFW_MOUSE_BUTTON_3){
        dragging = (action==GLFW_PRESS)?true:false;
        return true;
    }
    return false;

}

template<class camera_t>
bool Controlable_Camera<camera_t>::scroll_event(GLFWwindow* window, double xoffset, double yoffset){
    return false;
}
