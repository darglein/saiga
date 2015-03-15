#pragma once

#include <libhello/util/glm.h>
#include <libhello/glfw/glfw_eventhandler.h>



template<typename camera_t>
class Controllable_Camera : public glfw_KeyListener , public glfw_MouseListener{
private:
    bool dragging = false;
    double lastmx=0,lastmy=0;
    int FORWARD=0,RIGHT=0;
    vec3 positionAtUpdate;

public:
    camera_t* cam;
    float movementSpeed = 1;
    float rotationSpeed = 1;
    Controllable_Camera(camera_t* cam):cam(cam){
        glfw_EventHandler::addKeyListener(this);glfw_EventHandler::addMouseListener(this);
    }
    void update(float delta);
    void predictInterpolate(float interpolation);
    void setPosition(glm::vec3 cords);


    //glfw events
    bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods);
    bool character_event(GLFWwindow* window, unsigned int codepoint);
    bool cursor_position_event(GLFWwindow* window, double xpos, double ypos);
    bool mouse_button_event(GLFWwindow* window, int button, int action, int mods);
    bool scroll_event(GLFWwindow* window, double xoffset, double yoffset);


};

template<class camera_t>
void Controllable_Camera<camera_t>::setPosition(glm::vec3 cords)
{
    cam->setPosition(cords);
    cam->calculateModel();
    positionAtUpdate = vec3(cords);
}

template<class camera_t>
void Controllable_Camera<camera_t>::update(float delta){
    setPosition(positionAtUpdate);
    vec3 trans = delta*movementSpeed*FORWARD*vec3(0,0,-1) + delta*movementSpeed*RIGHT*vec3(1,0,0);
    cam->translateLocal(trans);
    cam->calculateModel();
    cam->updateFromModel();
    positionAtUpdate =vec3(cam->getPosition());
}

template<class camera_t>
void Controllable_Camera<camera_t>::predictInterpolate(float interpolation){
    vec3 interpolationTrans = (float)(1.0/60.0) * interpolation*movementSpeed*FORWARD*vec3(0,0,-1) + (float)(1.0/60.0) *interpolation*movementSpeed*RIGHT*vec3(1,0,0);
    cam->setPosition(positionAtUpdate);
    cam->translateLocal(interpolationTrans);
    cam->calculateModel();
    cam->updateFromModel();
}


template<class camera_t>
bool Controllable_Camera<camera_t>::key_event(GLFWwindow* window, int key, int scancode, int action, int mods){
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
bool Controllable_Camera<camera_t>::character_event(GLFWwindow* window, unsigned int codepoint){
    return false;
}

template<class camera_t>
bool Controllable_Camera<camera_t>::cursor_position_event(GLFWwindow* window, double xpos, double ypos){

    if(dragging){
        double relx = lastmx-xpos;
        double rely = lastmy-ypos;
        cam->turn((float)relx*rotationSpeed,(float)rely*rotationSpeed);
        cam->calculateModel();
        cam->updateFromModel();
    }
    lastmx = xpos;
    lastmy = ypos;
    return false;
}

template<class camera_t>
bool Controllable_Camera<camera_t>::mouse_button_event(GLFWwindow* window, int button, int action, int mods){
    if(button==GLFW_MOUSE_BUTTON_3){
        dragging = (action==GLFW_PRESS)?true:false;
        return true;
    }
    return false;

}

template<class camera_t>
bool Controllable_Camera<camera_t>::scroll_event(GLFWwindow* window, double xoffset, double yoffset){
    return false;
}
