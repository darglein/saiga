#include "saiga/util/joystick.h"
#include "saiga/util/assert.h"
#include <iostream>
Joystick2 joystick;

Joystick2::Joystick2() : Keyboard(50)
{

}

void Joystick2::setCount(int _axisCount, int _buttonCount)
{
    axisCount = _axisCount;
    buttonCount = _buttonCount;
    virtualButtonCount = buttonCount + axisCount*2;
    axis.resize(axisCount);
    keystate.resize(virtualButtonCount);
}

void Joystick2::setAxisState(int ax, float state)
{
    if(ax >= 0 && ax < (int)axis.size()){
       assert(state >= -1 && state <= 1);
        axis[ax] = state;
    }
}

int Joystick2::setVirtualAxisKeyState(int ax, float state)
{
    int key0 = buttonCount + 2*ax;
    int key1 = buttonCount + 2*ax + 1;
    int state0 = state > virtualButtonThreshold;
    int state1 = state < -virtualButtonThreshold;

    int r0 = setKeyState(key0,state0);
    int r1 = setKeyState(key1,state1);

    if(r0!=-1 || r1!=-1)
        return key0;

    return -1;
}

float Joystick2::getAxisState(int key)
{
    assert(key >= 0 && key < (int)axis.size());
    return axis[key];
}

float Joystick2::getMappedAxisState(int mappedKey, const std::vector<int> &keymap)
{
    assert(mappedKey >= 0 && mappedKey < (int)keymap.size());
    int key = keymap[mappedKey];
    assert(key >= 0 && key < (int)axis.size());
    return axis[key];
}

void Joystick2::printAxisState()
{
    std::cout << "[";
    for(auto k : axis){
        std::cout << k << ",";
    }
    std::cout << "]" << std::endl;
}
