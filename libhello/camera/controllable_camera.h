#pragma once

#include <libhello/util/glm.h>
#include <libhello/glfw/glfw_eventhandler.h>
#include <array>


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
    int keyForward = GLFW_KEY_UP;
    int keyRight = GLFW_KEY_RIGHT;
    int keyLeft = GLFW_KEY_LEFT;
    int keyBackwards = GLFW_KEY_DOWN;
    enum Key{
        Forward = 0,
        Backward = 1,
        Left = 2,
        Right =3
    };

    std::array<bool,4> keyPressed;

    int buttonDrag = GLFW_MOUSE_BUTTON_3;

    Controllable_Camera(camera_t* cam):cam(cam){
    }

    void update(float delta);
    void predictInterpolate(float interpolation);
    void setPosition(const glm::vec3 &cords);

    void enableInput();
    void disableInput();

    //glfw events
    bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods);
    bool character_event(GLFWwindow* window, unsigned int codepoint);
    bool cursor_position_event(GLFWwindow* window, double xpos, double ypos);
    bool mouse_button_event(GLFWwindow* window, int button, int action, int mods);
    bool scroll_event(GLFWwindow* window, double xoffset, double yoffset);


};

template<class camera_t>
void Controllable_Camera<camera_t>::enableInput()
{
    glfw_EventHandler::addKeyListener(this,0);
    glfw_EventHandler::addMouseListener(this,0);
}

template<class camera_t>
void Controllable_Camera<camera_t>::disableInput()
{
    for (int i = 0; i < keyPressed.size(); ++i){
        keyPressed[i] = false;
    }
    glfw_EventHandler::removeKeyListener(this);
    glfw_EventHandler::removeKeyListener(this);
}

template<class camera_t>
void Controllable_Camera<camera_t>::setPosition(const glm::vec3& cords)
{
    cam->setPosition(cords);
    cam->calculateModel();
    positionAtUpdate = vec3(cords);
}

template<class camera_t>
void Controllable_Camera<camera_t>::update(float delta){
    FORWARD =  keyPressed[Forward] - keyPressed[Backward];
    RIGHT = keyPressed[Right] - keyPressed[Left];

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
    if (key == keyForward){
        keyPressed[Forward] = action!=GLFW_RELEASE;
        return true;
    }  else if (key == keyBackwards){
        keyPressed[Backward] = action!=GLFW_RELEASE;
        return true;
    }else if (key == keyRight){
        keyPressed[Right] = action!=GLFW_RELEASE;
        return true;
    }else if (key == keyLeft){
        keyPressed[Left] = action!=GLFW_RELEASE;
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
    if(button==buttonDrag){
        dragging = (action==GLFW_PRESS)?true:false;
        return true;
    }
    return false;

}

template<class camera_t>
bool Controllable_Camera<camera_t>::scroll_event(GLFWwindow* window, double xoffset, double yoffset){
    return false;
}
