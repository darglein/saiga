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
std::vector<glfw_JoystickListener*> glfw_EventHandler::joystickListener;
std::vector<glfw_KeyListener*> glfw_EventHandler::keyListener;
std::vector<glfw_MouseListener*> glfw_EventHandler::mouseListener;
std::vector<glfw_ResizeListener*> glfw_EventHandler::resizeListener;



glfw_JoystickListener::glfw_JoystickListener()
{
    glfw_EventHandler::joystickListener.push_back(this);
}

glfw_JoystickListener::~glfw_JoystickListener()
{
    auto& v = glfw_EventHandler::joystickListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}


glfw_KeyListener::glfw_KeyListener()
{
    glfw_EventHandler::keyListener.push_back(this);
}

glfw_KeyListener::~glfw_KeyListener()
{
    auto& v = glfw_EventHandler::keyListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}

glfw_MouseListener::glfw_MouseListener()
{
    glfw_EventHandler::mouseListener.push_back(this);
}

glfw_MouseListener::~glfw_MouseListener()
{
    auto& v = glfw_EventHandler::mouseListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}

glfw_ResizeListener::glfw_ResizeListener()
{
    glfw_EventHandler::resizeListener.push_back(this);
}

glfw_ResizeListener::~glfw_ResizeListener()
{
    auto& v = glfw_EventHandler::resizeListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}


void glfw_EventHandler::joystick_key_callback(int button, bool pressed)
{
    for (auto& rl : joystickListener)
    {
        if (rl->joystick_event(button, pressed)) return;
    }
}

void glfw_EventHandler::window_size_callback(GLFWwindow* window, int width, int height)
{
    for (auto& rl : resizeListener)
    {
        if (rl->window_size_callback(window, width, height)) return;
    }
}

void glfw_EventHandler::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    mouse.setPosition(ivec2(xpos, ypos));

    for (auto& ml : mouseListener)
    {
        if (ml->cursor_position_event(window, xpos, ypos)) return;
    }
}

void glfw_EventHandler::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_RELEASE) mouse.setKeyState(button, action == GLFW_PRESS);

    for (auto& ml : mouseListener)
    {
        if (ml->mouse_button_event(window, button, action, mods)) return;
    }
}

void glfw_EventHandler::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    for (auto& ml : mouseListener)
    {
        if (ml->scroll_event(window, xoffset, yoffset)) return;
    }
}

void glfw_EventHandler::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_RELEASE) keyboard.setKeyState(key, action == GLFW_PRESS);

    for (auto& kl : keyListener)
    {
        if (kl->key_event(window, key, scancode, action, mods)) return;
    }
}

void glfw_EventHandler::character_callback(GLFWwindow* window, unsigned int codepoint)
{
    for (auto& kl : keyListener)
    {
        if (kl->character_event(window, codepoint)) return;
    }
}

}  // namespace Saiga
