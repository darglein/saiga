#pragma once

#include <saiga/config.h>
#include "saiga/window/window.h"
#include "saiga/glfw/glfw_eventhandler.h"
#include "saiga/glfw/glfw_joystick.h"

#include <array>
#include <string>
#include <mutex>
#include <memory>
#include <list>
#include <thread>

struct GLFWwindow;
struct GLFWcursor;
class Image;


class SAIGA_GLOBAL glfw_Window : public OpenGLWindow, public glfw_ResizeListener{
public:
    GLFWwindow* window = nullptr;

    bool recordingVideo = false;

    glfw_Window(WindowParameters windowParameters);
    virtual ~glfw_Window();


    void setCursorPosition(int x, int y);
    void showMouseCursor();
    void hideMouseCursor();
    void disableMouseCursor();
    void setGLFWcursor(GLFWcursor* cursor);
    GLFWcursor* createGLFWcursor(Image* image, int midX, int midY);
    void setWindowIcon(Image *image);

    virtual bool window_size_callback(GLFWwindow* window, int width, int height) override;
protected:
    virtual bool initWindow() override;
    virtual bool initInput() override;
    virtual bool shouldClose() override;
    virtual void checkEvents() override;
    virtual void swapBuffers() override;
    virtual void freeContext() override;
public:
    //static glfw stuff
    static void error_callback(int error, const char* description);
    static bool initGlfw();
    static void getCurrentPrimaryMonitorResolution(int *width, int *height);
    static void getMaxResolution(int *width, int *height);
public:
    //TODO: remove everything from here

    using OpenGLWindow::startMainLoop;
    void startMainLoopNoRender(float ticksPerSecond);
    void screenshotParallelWrite(const std::string &file);
    void setVideoRecordingLimit(int limit){queueLimit = limit;}
private:

    int currentScreenshot = 0;
    std::string parallelScreenshotPath;

    std::list<std::shared_ptr<Image>> queue;
    std::mutex lock;
    bool ssRunning = false;
    int queueLimit = 200;

    bool waitForWriters = false;

#define WRITER_COUNT 7
    std::thread* sswriterthreads[WRITER_COUNT];
    void processScreenshots();
};
