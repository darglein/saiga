#pragma once

#include <saiga/config.h>
#include <array>


enum class JoystickButton : int{
    Confirm = 0, //A
    Back = 1,    //B
    Unused1 = 2, //X
    Unused2 = 3, //Y
    Lookahead = 4,
    Grenade = 5,
    Unused4 = 6,
    Unused5 = 7,
    Unused6 = 8,
    Unused7 = 9,
    Up = 10,
    Right = 11,
    Down = 12,
    Left = 13
};

struct SAIGA_GLOBAL Joystick{
    bool enabled(){return joystickId != -1;}
    int joystickId = -1;

    float moveX = 0.f;
    float moveY = 0.f;
    float aimX = 0.f;
    float aimY = 0.f;
    float fire = 0.f;

    inline bool buttonPressed(JoystickButton b){return buttonsPressed[(int)b];}

    void getCurrentStateFromGLFW();

    void enableFirstJoystick();

    void joystick_callback(int joy, int event);

private:
    std::array<bool, 14> buttonsPressed = {{}};
    void checkButton(JoystickButton b, const unsigned char *ax);
};
