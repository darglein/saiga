#pragma once


#include "libhello/window/window.h"
#include <array>



struct GLFWwindow;
struct GLFWcursor;
class Image;

struct Joystick{
    bool enabled(){return joystickId != -1;}
    int joystickId = -1;

    float moveX = 0.f;
    float moveY = 0.f;
    float aimX = 0.f;
    float aimY = 0.f;
    float fire = 0.f;

    enum Buttons{
        Confirm,
        Back,
        Left,
        Right,
        Up,
        Down,
        Lookahead
    };

    std::array<bool, 7> buttonsPressed = {{}};

    void getCurrentStateFromGLFW();
};

class glfw_Window : public Window{
protected:
    GLFWwindow* window = nullptr;

    double timeScale = 1.f;

    bool initWindow();
    bool initInput();
public:

    Joystick joystick;

    double lastSwapBuffersMS = 0;
    double lastPolleventsMS = 0;

    glfw_Window(const std::string &name,int window_width,int window_height, bool fullscreen);
    virtual ~glfw_Window();

    static bool initGlfw();

    static void getCurrentPrimaryMonitorResolution(int *width, int *height);

    static void getMaxResolution(int *width, int *height);
    void showMouseCursor();
    void hideMouseCursor();
    void disableMouseCursor();

    void close();
    void startMainLoop();
    void startMainLoopConstantUpdateRenderInterpolation(int ticksPerSecond);
    void startMainLoopNoRender(float ticksPerSecond);
    void setTimeScale(double timeScale);




    static void error_callback(int error, const char* description);


    void setGLFWcursor(GLFWcursor* cursor);
    GLFWcursor* createGLFWcursor(Image* image, int midX, int midY);
};
