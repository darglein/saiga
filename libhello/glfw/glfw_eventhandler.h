#pragma once

#include <GLFW/glfw3.h>
#include <vector>

class glfw_KeyListener{
public:
    virtual bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods) = 0;
    virtual bool character_event(GLFWwindow* window, unsigned int codepoint) = 0;
};

class glfw_MouseListener{
public:
    virtual bool cursor_position_event(GLFWwindow* window, double xpos, double ypos) = 0;
    virtual bool mouse_button_event(GLFWwindow* window, int button, int action, int mods) = 0;
    virtual bool scroll_event(GLFWwindow* window, double xoffset, double yoffset) = 0;
};

class glfw_EventHandler{
private:

    template<typename T>
    struct Listener{
        T* listener = nullptr;
        int priority = 0;
        Listener(T* listener,int priority):listener(listener),priority(priority){}
    };
    static std::vector<Listener<glfw_KeyListener>> keyListener;
    static std::vector<Listener<glfw_MouseListener>> mouseListener;
public:
    static void addKeyListener(glfw_KeyListener* kl, int priority = 0);
    static void addMouseListener(glfw_MouseListener* ml, int priority = 0);

    static void removeKeyListener(glfw_KeyListener* kl, int priority = 0);
    static void removeMouseListener(glfw_MouseListener* ml, int priority = 0);

public:
    //these functions will be called by glfw from the method glfwPollEvents();
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void character_callback(GLFWwindow* window, unsigned int codepoint);
};

