#include "glfw/glfw_window.h"
#include "rendering/deferred_renderer.h"
#include "util/inputcontroller.h"
#include <chrono>
#include "libhello/util/error.h"
glfw_Window::glfw_Window(const std::string &name, int window_width, int window_height, bool fullscreen):Window(name,window_width,window_height, fullscreen)
{
}

glfw_Window::~glfw_Window()
{
    if(!window)
        return;
//    cout<<"~glfw_Window"<<endl;
    glfwDestroyWindow(window);
    glfwTerminate();

}

void glfw_Window::getNativeResolution(int* width, int *height)
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
    /* Initialize the library */
    if (!glfwInit())
        return false;

    return true;
}

bool glfw_Window::initWindow()
{
    //glfwInit has to be called before

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    //    glfwWindowHint(GLFW_SRGB_CAPABLE,1);




    GLFWmonitor* primary = glfwGetPrimaryMonitor();

    /* Create a windowed mode window and its OpenGL context */
    if (fullscreen){
        window = glfwCreateWindow(window_width, window_height, name.c_str(), primary, NULL);
    } else {
        window = glfwCreateWindow(window_width, window_height, name.c_str(), NULL, NULL);
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
    glfwSwapInterval(vsync ? 1 : 0);

    //Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum glewError = glewInit();
    if( glewError != GLEW_OK ){
        printf( "Error initializing GLEW! %s\n", glewGetErrorString( glewError ) );
    }

    glGetError(); //ignore first gl error after glew init

    Error::quitWhenError("initWindow()");

    return true;
}

bool glfw_Window::initInput(){
    //mouse
    glfwSetCursorPosCallback(window,glfw_EventHandler::cursor_position_callback);
    glfwSetMouseButtonCallback(window, glfw_EventHandler::mouse_button_callback);
    glfwSetScrollCallback(window, glfw_EventHandler::scroll_callback);
    //keyboard
    glfwSetCharCallback(window, glfw_EventHandler::character_callback);
    glfwSetKeyCallback(window, glfw_EventHandler::key_callback);

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
    /* Loop until the user closes the window */
    while (running && !glfwWindowShouldClose(window))
    {
        /* Render here */
        //        eventHandler.update();
        //        running &= !eventHandler.shouldQuit();

        update(1.0/60.0);

        renderer->render_intern();
        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
}

long long getTicksMS(){
    using namespace std::chrono;

    return  duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

void glfw_Window::startMainLoopConstantUpdateRenderInterpolation(int ticksPerSecond){
    const long long SKIP_TICKS_NORMAL_TIME = 1000000 / ticksPerSecond;
    long long SKIP_TICKS = SKIP_TICKS_NORMAL_TIME;

    setTimeScale(1.0);

    const int MAX_FRAMESKIP = 20;

    long long next_game_tick = getTicksMS();

    running = true;
    while( running && !glfwWindowShouldClose(window) ) {

        int loops = 0;
        while( getTicksMS() > next_game_tick ) {
            if (loops > MAX_FRAMESKIP){
                cout << "<Gameloop> Warning: Update loop is falling behind. (" << (getTicksMS() - next_game_tick)/1000 << "ms)" << endl;
                break;
            }
            update(1.0/60.0);


            SKIP_TICKS = ((double)SKIP_TICKS_NORMAL_TIME)/timeScale;

            next_game_tick += SKIP_TICKS;

            ++loops;
        }

        renderer->render_intern( ((float)(getTicksMS() + SKIP_TICKS - next_game_tick ))/ (float) (SKIP_TICKS ) );

        double now = glfwGetTime()*1000;

        glfwSwapBuffers(window);
        double now2 = glfwGetTime()*1000;
        lastSwapBuffersMS = now2 - now;

        /* Poll for and process events */
        glfwPollEvents();
        lastPolleventsMS = glfwGetTime()*1000 - now2;
    }
}

void glfw_Window::setTimeScale(double timeScale)
{
    this->timeScale = timeScale;
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
    if(image->bitDepth != 8 || image->channels != 4){
        cout<<"glfw_Window::createGLFWcursor(Image *image): image has the wrong format."<<endl;
        cout<<"Required format: RGBA8"<<endl;
    }


    GLFWimage glfwimage;
    glfwimage.width = image->width;
    glfwimage.height = image->height;
    glfwimage.pixels = image->data;

    return glfwCreateCursor(&glfwimage, midX, midY);
}

