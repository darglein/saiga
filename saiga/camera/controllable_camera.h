#pragma once

#include <saiga/opengl/opengl.h>
#include <saiga/camera/camera.h>
#include <GLFW/glfw3.h>

#include <saiga/util/glm.h>
#include <saiga/glfw/glfw_eventhandler.h>
#include <array>

static int maxCamId = 0;


template<typename camera_t>
class Controllable_Camera : public camera_t, public glfw_KeyListener , public glfw_MouseListener{
private:
    bool dragging = false;
    double lastmx=0,lastmy=0;
    int FORWARD=0,RIGHT=0;
    vec3 positionAtUpdate;

    bool recursive = false;
    int camId;

public:
    float movementSpeed = 1;
    float movementSpeedFast = 5;

    float rotationSpeed = 1;
    int keyForward = GLFW_KEY_UP;
    int keyRight = GLFW_KEY_RIGHT;
    int keyLeft = GLFW_KEY_LEFT;
    int keyBackwards = GLFW_KEY_DOWN;
    int keyUp = GLFW_KEY_SPACE;
    int keyFast = GLFW_KEY_LEFT_SHIFT;
    enum Key{
        Forward = 0,
        Backward = 1,
        Left = 2,
        Right = 3,
        Fast = 4,
        Up =5
    };

    std::array<bool,6> keyPressed {};

    int buttonDrag = GLFW_MOUSE_BUTTON_3;

    Controllable_Camera(){
//        cout << "Controllable_Camera() "<< endl;
        positionAtUpdate  = this->getPosition();
        this->rot = glm::quat_cast(this->model);

        camId = maxCamId++;
    }

    ~Controllable_Camera(){
//        cout << "~Controllable_Camera" << endl;
        disableInput();
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
//    cout << "ENABLE camera input " << camId << endl;

}

template<class camera_t>
void Controllable_Camera<camera_t>::disableInput()
{
    for (int i = 0; i < keyPressed.size(); ++i){
        keyPressed[i] = false;
    }
    glfw_EventHandler::removeKeyListener(this);
    glfw_EventHandler::removeMouseListener(this);
//    cout << "DISABLE camera input " << camId  << endl;

}

template<class camera_t>
void Controllable_Camera<camera_t>::setPosition(const glm::vec3& cords)
{
    camera_t::setPosition(cords);
    this->calculateModel();
    positionAtUpdate = vec3(cords);
}

template<class camera_t>
void Controllable_Camera<camera_t>::update(float delta){
    FORWARD =  keyPressed[Forward] - keyPressed[Backward];
    RIGHT = keyPressed[Right] - keyPressed[Left];

    float speed;
    if (keyPressed[Fast]){
        speed = movementSpeedFast;
    } else {
        speed = movementSpeed;
    }

    setPosition(positionAtUpdate);
    vec3 trans = delta*speed*FORWARD*vec3(0,0,-1) + delta*speed*RIGHT*vec3(1,0,0);
    vec3 transg =  vec3(0,1,0) * (delta*speed*keyPressed[Up]);
    this->translateLocal(trans);
    this->translateGlobal(transg);
    this->calculateModel();
    this->updateFromModel();
    positionAtUpdate =vec3(this->getPosition());
}

template<class camera_t>
void Controllable_Camera<camera_t>::predictInterpolate(float interpolation){
    vec3 interpolationTrans = (float)(1.0/60.0) * interpolation*movementSpeed*FORWARD*vec3(0,0,-1) + (float)(1.0/60.0) *interpolation*movementSpeed*RIGHT*vec3(1,0,0);
    this->setPosition(positionAtUpdate);
    this->translateLocal(interpolationTrans);
    this->calculateModel();
    this->updateFromModel();
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
    }else if (key == keyFast){
        keyPressed[Fast] = action!=GLFW_RELEASE;
        return true;
    }else if (key == keyUp){
        keyPressed[Up] = action!=GLFW_RELEASE;
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
    //prevent recursive call
    if (recursive){
        recursive = false;
        return false;
    }


    if(dragging){
//        cout << "drag " << camId << " " << xpos << "  " << ypos << endl;
        double relx = lastmx-xpos;
        double rely = lastmy-ypos;
        this->turn((float)relx*rotationSpeed,(float)rely*rotationSpeed);
        this->calculateModel();
        this->updateFromModel();

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
