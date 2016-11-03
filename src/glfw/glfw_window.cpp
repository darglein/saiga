#include "saiga/glfw/glfw_window.h"

#include <saiga/opengl/opengl.h>
#include "saiga/glfw/glfw_eventhandler.h"
#include <GLFW/glfw3.h>

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/rendering/renderer.h"

#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/util/inputcontroller.h"
#include <chrono>
#include "saiga/util/error.h"
#include "saiga/framework.h"





glfw_Window::glfw_Window(WindowParameters windowParameters):
    Window(windowParameters)
{
}

glfw_Window::~glfw_Window()
{
    if(!window)
        return;

    if (ssRunning){
        ssRunning = false;

        for (int i = 0; i < WRITER_COUNT; ++i){
            sswriterthreads[i]->join();
            delete sswriterthreads[i];
        }
    }


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
        return false;
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

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    //    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    //    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    //    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	if (windowParameters.coreContext) {
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL
    }
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, windowParameters.debugContext);
    //    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    //    glfwWindowHint(GLFW_SRGB_CAPABLE,1);

    glfwWindowHint(GLFW_DECORATED,!windowParameters.borderLess());
    glfwWindowHint(GLFW_FLOATING,windowParameters.alwaysOnTop);
    glfwWindowHint(GLFW_RESIZABLE,windowParameters.resizeAble);

	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
	// GLFW_REFRESH_RATE, GLFW_DONT_CARE = highest
	glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);

    std::cout << "Creating GLFW Window. " << getWidth() << "x" << getHeight() <<
                 " Fullscreen=" << windowParameters.fullscreen() <<
                 " Borderless=" << windowParameters.borderLess() <<
                 std::endl;


    switch (windowParameters.mode){
    case WindowParameters::Mode::windowed:
        window = glfwCreateWindow(getWidth(), getHeight(), getName().c_str(), NULL, NULL);
        break;
    case WindowParameters::Mode::fullscreen:
        window = glfwCreateWindow(getWidth(), getHeight(), getName().c_str(), monitor, NULL);
        break;
    case WindowParameters::Mode::borderLessWindowed:
        window = glfwCreateWindow(getWidth(), getHeight(), getName().c_str(), NULL, NULL);
        break;
    case WindowParameters::Mode::borderLessFullscreen:
#ifndef WIN32
        std::cerr << "Windowed Fullscreen may not work on your system." << std::endl;
#endif

        //works in windows 7. Does not work in ubuntu with gnome
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		
        window = glfwCreateWindow(getWidth(), getHeight(), getName().c_str(), NULL, NULL);

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

    glfwMakeContextCurrent(window);

    //vsync
    glfwSwapInterval(windowParameters.vsync ? 1 : 0);


    //framebuffer size != window size
    glfwGetFramebufferSize(window, &windowParameters.width, &windowParameters.height);


    //not needed but makes start cleaner
    glfwPollEvents();
    glfwSwapBuffers(window);


    if (windowParameters.updateJoystick){
        glfwSetJoystickCallback(glfw_Joystick::joystick_callback);
        glfw_Joystick::enableFirstJoystick();
    }


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

    IC.add("quit", [this](ICPARAMS){(void)args;this->close();});

    return true;
}



void glfw_Window::freeContext()
{

    //Disable text input
    //    SDL_StopTextInput();

    glfwTerminate();
}


void glfw_Window::startMainLoopNoRender(float ticksPerSecond)
{
    const float dt = 1.0f/ticksPerSecond;
//    setTimeScale(1.0);

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
        assert(0);
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
    glfwSwapBuffers(window);
}

void glfw_Window::checkEvents()
{
    glfwPollEvents();
    if(windowParameters.updateJoystick)
        glfw_Joystick::update();
}


void glfw_Window::setWindowIcon(Image* image){
    assert(window);
    if(image->Format().getBitDepth() != 8 || image->Format().getChannels() != 4){
        cout<<"glfw_Window::setWindowIcon(Image *image): image has the wrong format."<<endl;
        cout<<"Required format: RGBA8"<<endl;
        assert(0);
    }


    GLFWimage glfwimage;
    glfwimage.width = image->width;
    glfwimage.height = image->height;
    glfwimage.pixels = image->getRawData();

    //only works with glfw version 3.2 and up
    glfwSetWindowIcon(window,1,&glfwimage);
}


void glfw_Window::screenshotParallelWrite(const std::string &file){

    if (currentScreenshot == 0){
        cout<<"Starting " << WRITER_COUNT << " screenshot writers" <<file<<endl;
        for (int i = 0; i < WRITER_COUNT; ++i){
            sswriterthreads[i] = new std::thread(&glfw_Window::processScreenshots, this);
        }
        ssRunning = true;
    }


    int w = renderer->width;
    int h = renderer->height;

    std::shared_ptr<Image> img = std::make_shared<Image>();
    img->width = w;
    img->height = h;
    img->Format() = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
    img->create();

    auto tex = getRenderer()->postProcessor.getCurrentTexture();
    tex->bind();
    glGetTexImage(tex->getTarget(),0,GL_RGB,GL_UNSIGNED_BYTE,img->getRawData());
    tex->unbind();



    if (waitForWriters){

        while(true){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            lock.lock();
            if (queue.size() < 5){
                lock.unlock();
                break;
            }
            lock.unlock();
        }

        waitForWriters = false;
    }
    lock.lock();
    parallelScreenshotPath = file;
    queue.push_back(img);


    if ((int)queue.size() > queueLimit){ //one frame full HD ~ 4.5Mb
        waitForWriters = true;
    }
//        cout << "queue size: " << queue.size() << endl;

    lock.unlock();
}


void glfw_Window::processScreenshots()
{

    while(ssRunning){
        int cur = 0;
        bool took = false;
        int queueSize = 0;
        lock.lock();
        std::shared_ptr<Image> f;
        if (!queue.empty()){
            f = queue.front();
            queueSize = queue.size();
            if (f){
                took = true;
                queue.pop_front();
                cur = currentScreenshot++;
            }
        }

        lock.unlock();

        if (took){
            long long start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            TextureLoader::instance()->saveImage(parallelScreenshotPath+ std::to_string(cur) + ".bmp",*f);
//            f->save(().c_str());
            cout << "write " << cur  << " (" <<queueSize << ") " << (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - start)/1000 << "ms"<< endl;


        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}




