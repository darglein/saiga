/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "glfw_eventhandler.h"

#include "internal/noGraphicsAPI.h"

#include <GLFW/glfw3.h>

#include "glfw_joystick.h"

namespace Saiga
{
std::vector<glfw_EventHandler::Listener<glfw_JoystickListener>> glfw_EventHandler::joystickListener;
std::vector<glfw_EventHandler::Listener<glfw_KeyListener>> glfw_EventHandler::keyListener;
std::vector<glfw_EventHandler::Listener<glfw_MouseListener>> glfw_EventHandler::mouseListener;
std::vector<glfw_EventHandler::Listener<glfw_ResizeListener>> glfw_EventHandler::resizeListener;


void glfw_EventHandler::addJoystickListener(glfw_JoystickListener* jl, int priority)
{
    addListener<glfw_JoystickListener>(joystickListener, jl, priority);
}

void glfw_EventHandler::removeJoystickListener(glfw_JoystickListener* jl)
{
    removeListener<glfw_JoystickListener>(joystickListener, jl);
}

void glfw_EventHandler::addKeyListener(glfw_KeyListener* kl, int priority)
{
    addListener<glfw_KeyListener>(keyListener, kl, priority);
}


void glfw_EventHandler::removeKeyListener(glfw_KeyListener* kl)
{
    removeListener<glfw_KeyListener>(keyListener, kl);
}


void glfw_EventHandler::addMouseListener(glfw_MouseListener* l, int priority)
{
    addListener<glfw_MouseListener>(mouseListener, l, priority);
}

void glfw_EventHandler::removeMouseListener(glfw_MouseListener* l)
{
    removeListener<glfw_MouseListener>(mouseListener, l);
}


void glfw_EventHandler::addResizeListener(glfw_ResizeListener* l, int priority)
{
    addListener<glfw_ResizeListener>(resizeListener, l, priority);
}

void glfw_EventHandler::removeResizeListener(glfw_ResizeListener* l)
{
    removeListener<glfw_ResizeListener>(resizeListener, l);
}

void glfw_EventHandler::joystick_key_callback(int button, bool pressed)
{
    //    std::cout<<"joystick_key_callback "<<(int)button<<" "<<pressed<<std::endl;

    // forward event to all listeners
    for (auto& rl : joystickListener)
    {
        if (rl.listener->joystick_event(button, pressed)) return;
    }
}

void glfw_EventHandler::window_size_callback(GLFWwindow* window, int width, int height)
{
    //    std::cout << "window_size_callback " << width << " " << height << std::endl;
    // forward event to all listeners
    for (auto& rl : resizeListener)
    {
        if (rl.listener->window_size_callback(window, width, height)) return;
    }
}

void glfw_EventHandler::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    mouse.setPosition(ivec2(xpos, ypos));

    // forward event to all listeners
    for (auto& ml : mouseListener)
    {
        if (ml.listener->cursor_position_event(window, xpos, ypos)) return;
    }
}

void glfw_EventHandler::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_RELEASE) mouse.setKeyState(button, action == GLFW_PRESS);

    // forward event to all listeners
    for (auto& ml : mouseListener)
    {
        if (ml.listener->mouse_button_event(window, button, action, mods)) return;
    }
}

void glfw_EventHandler::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // forward event to all listeners
    for (auto& ml : mouseListener)
    {
        if (ml.listener->scroll_event(window, xoffset, yoffset)) return;
    }
}

void glfw_EventHandler::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_RELEASE) keyboard.setKeyState(key, action == GLFW_PRESS);

    // forward event to all listeners
    for (auto& kl : keyListener)
    {
        if (kl.listener->key_event(window, key, scancode, action, mods)) return;
    }
}

void glfw_EventHandler::character_callback(GLFWwindow* window, unsigned int codepoint)
{
    // forward event to all listeners
    for (auto& kl : keyListener)
    {
        if (kl.listener->character_event(window, codepoint)) return;
    }
}

}  // namespace Saiga
