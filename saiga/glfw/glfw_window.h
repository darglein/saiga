#pragma once

#include <saiga/config.h>
#include "saiga/window/window.h"
#include "saiga/glfw/glfw_eventhandler.h"

#include <array>



struct GLFWwindow;
struct GLFWcursor;
class Image;

struct SAIGA_GLOBAL Joystick{
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


struct SAIGA_GLOBAL glfw_Window_Parameters{
    enum class Mode{
        windowed,
        fullscreen,
        borderLessWindowed,
        borderLessFullscreen
    };

    int width = 1600;
    int height = 900;

    Mode mode =  Mode::windowed;
    bool alwaysOnTop = false;
    bool resizeAble = true;
    bool vsync = true;
    bool updateJoystick = false;
    int monitorId = 0; //Important for fullscreen mode. 0 is always the primary monitor.

    bool borderLess(){ return mode==Mode::borderLessWindowed || mode==Mode::borderLessFullscreen;}
    bool fullscreen(){ return mode==Mode::fullscreen || mode==Mode::borderLessFullscreen;}

    void setMode(bool fullscreen, bool borderLess);
};

class SAIGA_GLOBAL glfw_Window : public Window, public glfw_ResizeListener{
protected:
    GLFWwindow* window = nullptr;
    glfw_Window_Parameters windowParameters;
    double timeScale = 1.f;

    bool initWindow();
    bool initInput();
public:

    Joystick joystick;


    double lastSwapBuffersMS = 0;
    double lastPolleventsMS = 0;

    glfw_Window(const std::string &name,glfw_Window_Parameters windowParameters);
    virtual ~glfw_Window();

    void showMouseCursor();
    void hideMouseCursor();
    void disableMouseCursor();

    void close();
    void startMainLoop();
    void startMainLoopConstantUpdateRenderInterpolation(int ticksPerSecond);
    void startMainLoopNoRender(float ticksPerSecond);
    void setTimeScale(double timeScale);

    virtual bool window_size_callback(GLFWwindow* window, int width, int height) override;


    void setGLFWcursor(GLFWcursor* cursor);
    GLFWcursor* createGLFWcursor(Image* image, int midX, int midY);



    static void error_callback(int error, const char* description);
    static bool initGlfw();
    static void getCurrentPrimaryMonitorResolution(int *width, int *height);
    static void getMaxResolution(int *width, int *height);
    void setWindowIcon(Image *image);
};
