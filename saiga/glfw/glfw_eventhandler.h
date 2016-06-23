#pragma once


#include <saiga/config.h>
#include <vector>
#include <algorithm>

struct GLFWwindow;
enum class JoystickButton;

class SAIGA_GLOBAL glfw_JoystickListener{
public:
    virtual ~glfw_JoystickListener(){}
    virtual bool joystick_event(JoystickButton button, bool pressed) = 0;
};

class SAIGA_GLOBAL glfw_KeyListener{
public:
    virtual ~glfw_KeyListener(){}
    virtual bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods) = 0;
    virtual bool character_event(GLFWwindow* window, unsigned int codepoint) = 0;
};

class SAIGA_GLOBAL glfw_MouseListener{
public:
    virtual ~glfw_MouseListener(){}
    virtual bool cursor_position_event(GLFWwindow* window, double xpos, double ypos) = 0;
    virtual bool mouse_button_event(GLFWwindow* window, int button, int action, int mods) = 0;
    virtual bool scroll_event(GLFWwindow* window, double xoffset, double yoffset) = 0;
};

class SAIGA_GLOBAL glfw_ResizeListener{
public:
    virtual ~glfw_ResizeListener(){}
    virtual bool window_size_callback(GLFWwindow* window, int width, int height) = 0;

};

class SAIGA_GLOBAL glfw_EventHandler{
private:

    template<typename T>
    struct Listener{
        T* listener = nullptr;
        int priority = 0;
        Listener(T* listener,int priority):listener(listener),priority(priority){}

        bool operator==(const Listener<T> &l1){
            return listener==l1.listener;
        }
    };
    static std::vector<Listener<glfw_JoystickListener>> joystickListener;
    static std::vector<Listener<glfw_KeyListener>> keyListener;
    static std::vector<Listener<glfw_MouseListener>> mouseListener;
    static std::vector<Listener<glfw_ResizeListener>> resizeListener;
private:
    template<typename T>
    static void addListener(std::vector<Listener<T>> &list, T* t, int priority = 0);
    template<typename T>
    static void removeListener(std::vector<Listener<T>> &list, T* t);
public:

    /**
     * @brief addJoysticklistener
     * Adds a Joystick Listener, does not add it, if it is already added
     */
    static void addJoystickListener(glfw_JoystickListener* jl, int priority = 0);
    static void removeJoystickListener(glfw_JoystickListener* kl);


    /**
     * @brief addKeyListener
     * Adds a Key Listener, does not add it, if it is already added
     */
    static void addKeyListener(glfw_KeyListener* kl, int priority = 0);
    static void removeKeyListener(glfw_KeyListener* kl);

    /**
     * @brief addMouseListener
     * Adds a Mouse Listener, does not add it, if it is already added
     */
    static void addMouseListener(glfw_MouseListener* ml, int priority = 0);
    static void removeMouseListener(glfw_MouseListener* ml);

    static void addResizeListener(glfw_ResizeListener* kl, int priority = 0);
    static void removeResizeListener(glfw_ResizeListener* kl);

public:
    //called from glfw window joystick //NOTE it is called inside the update step, not the glfwPollEvents function
    static void joystick_key_callback(JoystickButton button,bool pressed);

    static void window_size_callback(GLFWwindow* window, int width, int height);

    //these functions will be called by glfw from the method glfwPollEvents();
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void character_callback(GLFWwindow* window, unsigned int codepoint);
};


template<typename T>
void glfw_EventHandler::addListener(std::vector<Listener<T>> &list, T* t, int priority){
    Listener<T> l(t,priority);
    auto it = std::find(list.begin(),list.end(),l);

    if(it!=list.end())
        return;

    auto iter=list.begin();
    for(;iter!=list.end();++iter){
        if((*iter).priority<priority)
            break;
    }
    list.insert(iter,l);
}


template<typename T>
void glfw_EventHandler::removeListener(std::vector<Listener<T>> &list, T* t){
    auto it=list.begin();
    for(;it!=list.end();++it){
        if(it->listener==t){
            list.erase(it);
            return;
        }
    }
}
