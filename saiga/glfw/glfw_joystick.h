#pragma once

#include <saiga/util/joystick.h>
#include <saiga/config.h>
#include <array>


//enum class int : int{
//    Confirm = 0, //A
//    Back = 1,    //B
//    Unused1 = 2, //X
//    Unused2 = 3, //Y
//    Lookahead = 4,
//    Grenade = 5,
//    Unused4 = 6,
//    Unused5 = 7,
//    Unused6 = 8,
//    Unused7 = 9,
//    Up = 10,
//    Right = 11,
//    Down = 12,
//    Left = 13
//};

struct SAIGA_GLOBAL Joystick : public Keyboard{
    bool enabled(){return joystickId != -1;}
    int joystickId = -1;



    Joystick();

    void getCurrentStateFromGLFW();

    void enableFirstJoystick();

    void joystick_callback(int joy, int event);

};
