#include "saiga/glfw/glfw_joystick.h"
#include "glfw/glfw3.h"
#include <iostream>
#include "assert.h"
#include "saiga/glfw/glfw_eventhandler.h"

using std::cout;
using std::endl;

void Joystick::joystick_callback(int joy, int event)
{
    if (event == GLFW_CONNECTED)
    {
        // The joystick was connected
//        cout << "joystick was connected " << joy << endl;
        cout << "joystick was connected: " << glfwGetJoystickName(joy) << endl;
        joystickId = joy;
    }
    else if (event == GLFW_DISCONNECTED)
    {
        // The joystick was disconnected
        cout << "joystick was disconnected " << joy << endl;
        //glfwGetJoystickName would return nullptr here anyways
        if (joystickId == joy){
//            the enabled joystick was disconnected, take another one if one is enabled
            joystickId = -1;
            enableFirstJoystick();
        }

    }
}

void Joystick::enableFirstJoystick()
{
    if (joystickId != -1)
        return;

    for (int i = GLFW_JOYSTICK_1; i <= GLFW_JOYSTICK_LAST; ++i){
        if (glfwJoystickPresent(i)){
            //cout << "found joystick: " <<  i <<  ": " << glfwGetJoystickName(i) <<endl;

            //take first joystick
            cout << "using joystick: " <<  i <<endl;
            joystickId = i;
            break;
        }
    }
}

void Joystick::getCurrentStateFromGLFW()
{

    if (joystickId == -1){
        return;
    }
//    cout << "getCurrentStateFromGLFW " << joystickId << endl;

    int count;
    const float* axes = glfwGetJoystickAxes(joystickId, &count);

    //glfwGetJoystickAxes may call the joystick callback -> joystick may be disconnected now
    if (joystickId == -1)
        return;

    assert(count >= 6);
    if (count < 6)
        return;

//    cout << "count " << count << " joystickId "  << joystickId<< endl;

    //TODO button binding
//    moveX =  axes[0];
//    moveY = axes[1];
//    fire = axes[2];
//    aimX = axes[4];
//    aimY = axes[3];

    //the new glfw library version changed this button bindings somehow
    moveX =  axes[0];
    moveY = -axes[1];
    fire = -axes[5]; //left trigger: -axes[4];
    aimX = axes[2];
    aimY = -axes[3];


    int buttons;
    const unsigned char* ax = glfwGetJoystickButtons(joystickId, &buttons);

    //glfwGetJoystickButtons may call the joystick callback -> joystick may be disconnected now
    if (joystickId == -1)
        return;

    assert(buttons >= 14);
    if (buttons < 14)
        return;

    //TODO handle all buttons
    for (int i = 0; i < 14; ++i){
        checkButton(static_cast<JoystickButton>(i),ax);
    }
}

void Joystick::checkButton(JoystickButton b, const unsigned char* ax){
    bool old = buttonsPressed[static_cast<int>(b)];
    int newb = ax[static_cast<int>(b)];
    if (old != newb){
        glfw_EventHandler::joystick_key_callback(b,newb);
        buttonsPressed[static_cast<int>(b)] = newb;
    }
}
