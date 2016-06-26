#pragma once

#include "saiga/camera/controllable_camera.h"
#include <GLFW/glfw3.h>
#include <saiga/glfw/glfw_eventhandler.h>

template<typename camera_t>
class Glfw_Camera : public Controllable_Camera<camera_t>, public glfw_KeyListener , public glfw_MouseListener{
public:
    bool dragging = false;
    double lastmx=0,lastmy=0;
    bool recursive = false;


    int keyForward = GLFW_KEY_UP;
    int keyRight = GLFW_KEY_RIGHT;
    int keyLeft = GLFW_KEY_LEFT;
    int keyBackwards = GLFW_KEY_DOWN;
    int keyUp = GLFW_KEY_SPACE;
    int keyFast = GLFW_KEY_LEFT_SHIFT;

    int buttonDrag = GLFW_MOUSE_BUTTON_3;


    ~Glfw_Camera(){
        disableInput();
    }

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
void Glfw_Camera<camera_t>::enableInput()
{
    glfw_EventHandler::addKeyListener(this,0);
    glfw_EventHandler::addMouseListener(this,0);

}

template<class camera_t>
void Glfw_Camera<camera_t>::disableInput()
{
    for (int i = 0; i < this->keyPressed.size(); ++i){
        this->keyPressed[i] = false;
    }
    glfw_EventHandler::removeKeyListener(this);
    glfw_EventHandler::removeMouseListener(this);
}


template<class camera_t>
bool Glfw_Camera<camera_t>::key_event(GLFWwindow* window, int key, int scancode, int action, int mods){
    if (key == keyForward){
        this->keyPressed[this->Forward] = action!=GLFW_RELEASE;
        return true;
    }  else if (key == keyBackwards){
        this->keyPressed[this->Backward] = action!=GLFW_RELEASE;
        return true;
    }else if (key == keyRight){
        this->keyPressed[this->Right] = action!=GLFW_RELEASE;
        return true;
    }else if (key == keyLeft){
        this->keyPressed[this->Left] = action!=GLFW_RELEASE;
        return true;
    }else if (key == keyFast){
        this->keyPressed[this->Fast] = action!=GLFW_RELEASE;
        return true;
    }else if (key == keyUp){
        this->keyPressed[this->Up] = action!=GLFW_RELEASE;
        return true;
    }

    return false;
}


template<class camera_t>
bool Glfw_Camera<camera_t>::character_event(GLFWwindow* window, unsigned int codepoint){
    return false;
}

template<class camera_t>
bool Glfw_Camera<camera_t>::cursor_position_event(GLFWwindow* window, double xpos, double ypos){
    //prevent recursive call
    if (recursive){
        recursive = false;
        return false;
    }


    if(dragging){
        //        cout << "drag " << camId << " " << xpos << "  " << ypos << endl;
        double relx = lastmx-xpos;
        double rely = lastmy-ypos;
        this->mouseRotate(relx,rely);

        //prevent recursive call
        recursive = true;
        glfwSetCursorPos(window,lastmx, lastmy);
        xpos = lastmx;
        ypos = lastmy;
    }
    lastmx = xpos;
    lastmy = ypos;
    return false;
}

template<class camera_t>
bool Glfw_Camera<camera_t>::mouse_button_event(GLFWwindow* window, int button, int action, int mods){
    if(button==buttonDrag){
        dragging = (action==GLFW_PRESS)?true:false;

        return true;
    }
    return false;

}

template<class camera_t>
bool Glfw_Camera<camera_t>::scroll_event(GLFWwindow* window, double xoffset, double yoffset){
    return false;
}
