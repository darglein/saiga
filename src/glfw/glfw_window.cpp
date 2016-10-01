#include "saiga/glfw/glfw_window.h"

#include <saiga/opengl/opengl.h>
#include "saiga/glfw/glfw_eventhandler.h"
#include <GLFW/glfw3.h>

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/rendering/renderer.h"

#include "saiga/util/inputcontroller.h"
#include <chrono>
#include "saiga/util/error.h"
#include "saiga/framework.h"

//#define FORCEFRAMERATE 30
//#ifdef FORCEFRAMERATE
//#include <thread>
//#endif

Joystick* global_joystick = nullptr;

void joystick_callback_wrapper(int joy, int event)
{
    assert(global_joystick);
    global_joystick->joystick_callback(joy,event);
}

void glfw_Window_Parameters::setMode(bool fullscreen, bool borderLess)
{
    if(fullscreen){
        mode = (borderLess) ? Mode::borderLessFullscreen : Mode::fullscreen;
    }else{
        mode = (borderLess) ? Mode::borderLessWindowed : Mode::windowed;
    }
}





glfw_Window::glfw_Window(const std::string &name, glfw_Window_Parameters windowParameters):
    Window(name,windowParameters.width,windowParameters.height),windowParameters(windowParameters)
{
}

glfw_Window::~glfw_Window()
{
    if(!window)
        return;


    cleanupSaiga();

    glfwDestroyWindow(window);
    glfwTerminate();
	terminateOpenGL();
	cout << "GLFW: Terminated." << endl;

}

void glfw_Window::getCurrentPrimaryMonitorResolution(int *width, int *height)
{
    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

    cout << "Video Mode: " << mode->width << " x "<< mode->height << " @" << mode->refreshRate << "Hz" << endl;

    *width = mode->width;
    *height = mode->height;
}

void glfw_Window::getMaxResolution(int* width, int *height)
{
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    //get max video mode resolution
    int count;
    const GLFWvidmode* mode = glfwGetVideoModes(primary,&count);
    //    cout << "Video modes:" << endl;
    //    for (int i = 0; i < count; i++){
    //        cout << "Mode "<< i << ": " << mode[i].width << " x "<< mode[i].height << " @" << mode[i].refreshRate << "Hz" << endl;
    //    }

    cout << "Native Video Mode: " << mode[count-1].width << " x "<< mode[count-1].height << " @" << mode[count-1].refreshRate << "Hz" << endl;
    *width = mode[count-1].width;
    *height = mode[count-1].height;
}

void glfw_Window::hideMouseCursor()
{
    glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
}

void glfw_Window::showMouseCursor()
{
    glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_NORMAL);
}

void glfw_Window::disableMouseCursor()
{
    glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_DISABLED);
}

bool glfw_Window::initGlfw(){
	
    glfwSetErrorCallback(glfw_Window::error_callback);
	cout << "Initializing GLFW." << endl;
    /* Initialize the library */
    if (!glfwInit())
        return false;
	cout << "Initializing GLFW sucessfull!" << endl;
    return true;
}



bool glfw_Window::initWindow()
{
    if (!glfw_Window::initGlfw()){
        cout << "Could not initialize GLFW" << endl;
        return -1;
    }

    int monitorCount;
    GLFWmonitor** monitors = glfwGetMonitors(&monitorCount) ;
    windowParameters.monitorId = glm::clamp(windowParameters.monitorId,0,monitorCount-1);

    GLFWmonitor* monitor = monitors[windowParameters.monitorId];
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);


//    //don't allow other resolutions than the native monitor ones in fullscreen mode
    if(windowParameters.fullscreen()){
        windowParameters.width = mode->width;
        windowParameters.height = mode->height;
    }
    this->width = windowParameters.width;
    this->height = windowParameters.height;


    //glfwInit has to be called before

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    //    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    //    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    //    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL

//#if !defined(SAIGA_RELEASE)
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, windowParameters.debugContext);
//#endif
    //    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    //    glfwWindowHint(GLFW_SRGB_CAPABLE,1);

    glfwWindowHint(GLFW_DECORATED,!windowParameters.borderLess());
    glfwWindowHint(GLFW_FLOATING,windowParameters.alwaysOnTop);
    glfwWindowHint(GLFW_RESIZABLE,windowParameters.resizeAble);

	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
	// GLFW_REFRESH_RATE, GLFW_DONT_CARE = highest
	glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);

    std::cout << "Creating GLFW Window. " << width << "x" << height <<
                 " Fullscreen=" << windowParameters.fullscreen() <<
                 " Borderless=" << windowParameters.borderLess() <<
                 std::endl;


    switch (windowParameters.mode){
    case glfw_Window_Parameters::Mode::windowed:
        window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
        break;
    case glfw_Window_Parameters::Mode::fullscreen:
        window = glfwCreateWindow(width, height, name.c_str(), monitor, NULL);
        break;
    case glfw_Window_Parameters::Mode::borderLessWindowed:
        window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
        break;
    case glfw_Window_Parameters::Mode::borderLessFullscreen:
#ifndef WIN32
        std::cerr << "Windowed Fullscreen may not work on your system." << std::endl;
#endif

        //works in windows 7. Does not work in ubuntu with gnome
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		
		window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);

		//move to correct monitor
		int xpos, ypos;
		glfwGetMonitorPos(monitor, &xpos, &ypos);
		glfwSetWindowPos(window, xpos, ypos);
        break;
    }

    if (!window)
    {
        glfwTerminate();
        cerr<<"glfwCreateWindow returned false!"<<endl;
        return false;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    //    //vsync
    glfwSwapInterval(windowParameters.vsync ? 1 : 0);


    //framebuffer size != window size
    glfwGetFramebufferSize(window, &width, &height);


    //not needed but makes start cleaner
    glfwPollEvents();
    glfwSwapBuffers(window);

    global_joystick = &joystick;

    glfwSetJoystickCallback(joystick_callback_wrapper);
    joystick.enableFirstJoystick();

    return true;
}

bool glfw_Window::initInput(){
    glfwSetFramebufferSizeCallback(window, glfw_EventHandler::window_size_callback);

    //mouse
    glfwSetCursorPosCallback(window,glfw_EventHandler::cursor_position_callback);
    glfwSetMouseButtonCallback(window, glfw_EventHandler::mouse_button_callback);
    glfwSetScrollCallback(window, glfw_EventHandler::scroll_callback);
    //keyboard
    glfwSetCharCallback(window, glfw_EventHandler::character_callback);
    glfwSetKeyCallback(window, glfw_EventHandler::key_callback);

    glfw_EventHandler::addResizeListener(this,0);

    IC.add("quit", [this](ICPARAMS){(void)args;this->quit();});

    return true;
}



void glfw_Window::close()
{

    //Disable text input
    //    SDL_StopTextInput();

    glfwTerminate();
}



void glfw_Window::startMainLoop(){
    running = true;

    const long long SKIP_TICKS_NORMAL_TIME = 1000000 / 60.0;
    long long SKIP_TICKS = SKIP_TICKS_NORMAL_TIME;
    const float dt = 1.0f/60.0;

    long long next_game_tick = getGameTicks();

    /* Loop until the user closes the window */
    while (running && !glfwWindowShouldClose(window))
    {

        update(dt);

        //in case timescale was changed
        SKIP_TICKS = ((double)SKIP_TICKS_NORMAL_TIME)/timeScale;

        render(dt,0);

        if (recordingVideo)
            screenshotParallelWrite("screenshots/video/");

        long long durationTicks = getGameTicks() - next_game_tick;

        if (durationTicks < SKIP_TICKS){
//            cout << "sleeping for " << (SKIP_TICKS - durationTicks) << endl;

            //force framerate
           // std::this_thread::sleep_for(std::chrono::milliseconds((int)( (SKIP_TICKS - durationTicks)/1000)));
        }


        next_game_tick = getGameTicks();

        swapBuffers();
        checkEvents();

        assert_no_glerror_end_frame();
    }
}



void glfw_Window::startMainLoopConstantUpdateRenderInterpolation(int ticksPerSecond, int maxFrameSkip){
    const long long SKIP_TICKS_NORMAL_TIME = 1000000 / ticksPerSecond;
    long long SKIP_TICKS = SKIP_TICKS_NORMAL_TIME;
    const float dt = 1.0f/ticksPerSecond;

    setTimeScale(1.0);


    long long next_game_tick = getGameTicks();

    running = true;
    while( running && !glfwWindowShouldClose(window) ) {

        int loops = 0;
        while( getGameTicks() > next_game_tick ) {
            if (loops > maxFrameSkip){
//                cout << "<Gameloop> Warning: Update loop is falling behind. (" << (getTicksMS() - next_game_tick)/1000 << "ms)" << endl;
                break;
            }

            update(dt);


            //if this flag is set, drop some updates, NOTE: this will cause the game to run slower!
            if (gameloopDropAccumulatedUpdates){
                cout << "<Gameloop> Dropping accumulated updates." << endl;
                next_game_tick = getGameTicks();
                gameloopDropAccumulatedUpdates = false;
            }


            SKIP_TICKS = ((double)SKIP_TICKS_NORMAL_TIME)/timeScale;

            next_game_tick += SKIP_TICKS;

            ++loops;
        }
        float interpolation = glm::clamp(((float)(getGameTicks() + SKIP_TICKS - next_game_tick ))/ (float) (SKIP_TICKS ),0.0f,1.0f);
        render(dt,interpolation);



#ifdef FORCEFRAMERATE
        std::this_thread::sleep_for(std::chrono::milliseconds((int)( 1000.f/FORCEFRAMERATE)));
#endif

    swapBuffers();
    checkEvents();


        assert_no_glerror_end_frame();
    }
}

void glfw_Window::startMainLoopNoRender(float ticksPerSecond)
{
    const float dt = 1.0f/ticksPerSecond;
    setTimeScale(1.0);

    int simulatedTicks = 0;
    running = true;
    while( running && !glfwWindowShouldClose(window) ) {
        update(dt);

        renderer->renderer->interpolate(dt,0);

        checkEvents();

        simulatedTicks++;

//        if (simulatedTicks > 60 && ((int)(now2) % 5000) == 0){
//            cout << "<Gameloop> Simulated " << simulatedTicks  << "ticks (" << simulatedTicks*dt <<  "s)" << endl;
//            simulatedTicks = 0;
//        }

        assert_no_glerror_end_frame();
    }
}


bool glfw_Window::window_size_callback(GLFWwindow *window, int width, int height)
{
    (void)window;
    this->resize(width,height);
    return false;
}

void glfw_Window::error_callback(int error, const char* description){

    cout<<"glfw error: "<<error<<" "<<description<<endl;
}

void glfw_Window::setGLFWcursor(GLFWcursor *cursor)
{
    glfwSetCursor(window,cursor);
}


GLFWcursor* glfw_Window::createGLFWcursor(Image *image, int midX, int midY)
{
    if(image->Format().getBitDepth() != 8 || image->Format().getChannels() != 4){
        cout<<"glfw_Window::createGLFWcursor(Image *image): image has the wrong format."<<endl;
        cout<<"Required format: RGBA8"<<endl;
    }


    GLFWimage glfwimage;
    glfwimage.width = image->width;
    glfwimage.height = image->height;
    glfwimage.pixels = image->getRawData();

    return glfwCreateCursor(&glfwimage, midX, midY);
}

bool glfw_Window::shouldClose() {
    return glfwWindowShouldClose(window) || !running;
}

void glfw_Window::swapBuffers()
{
    double now = glfwGetTime()*1000;
    glfwSwapBuffers(window);
    double now2 = glfwGetTime()*1000;
    lastSwapBuffersMS = now2 - now;
}

void glfw_Window::checkEvents()
{
    double now2 = glfwGetTime()*1000;
    if(windowParameters.updateJoystick)
        joystick.getCurrentStateFromGLFW();
    glfwPollEvents();
    lastPolleventsMS = glfwGetTime()*1000 - now2;
}


void glfw_Window::setWindowIcon(Image* image){
    assert(window);
    if(image->Format().getBitDepth() != 8 || image->Format().getChannels() != 4){
        cout<<"glfw_Window::setWindowIcon(Image *image): image has the wrong format."<<endl;
        cout<<"Required format: RGBA8"<<endl;
    }


    GLFWimage glfwimage;
    glfwimage.width = image->width;
    glfwimage.height = image->height;
    glfwimage.pixels = image->getRawData();

    //only works with glfw version 3.2 and up
    glfwSetWindowIcon(window,1,&glfwimage);
}





