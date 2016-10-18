#include "saiga/glfw/glfw_joystick.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "saiga/util/assert.h"
#include "saiga/glfw/glfw_eventhandler.h"

using std::cout;
using std::endl;

Joystick::Joystick() : Keyboard(50)
{

}


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

    int aC, bC;
    const float* axes = glfwGetJoystickAxes(joystickId, &aC);
	if (joystickId == -1) {
		return;
	}
    const unsigned char* ax = glfwGetJoystickButtons(joystickId, &bC);
	if (joystickId == -1) {
		return;
	}
    joystick.setCount(aC,bC);

    for(int i = 0 ; i < aC ; ++i){
		float state = glm::clamp(axes[i],-1.0f,1.0f);
        joystick.setAxisState(i, state);
        int changed = joystick.setVirtualAxisKeyState(i, state);
        if(changed != -1){
//           cout << "changed " << i << " " << changed << " " << axes[i] << endl;

            glfw_EventHandler::joystick_key_callback(changed,joystick.getKeyState(changed));
            glfw_EventHandler::joystick_key_callback(changed+1,joystick.getKeyState(changed+1));
        }
    }

    for(int i = 0 ; i < bC ; ++i){
        int state = (int)ax[i] == GLFW_PRESS;
        int changed = joystick.setKeyState(i,state);
        if(changed != -1){
//            cout << "changed " << i << " " << state << endl;
            glfw_EventHandler::joystick_key_callback(i,state);
        }
    }
 //   joystick.printAxisState();
 //  joystick.printKeyState();

}


