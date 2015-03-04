#include "glfw/glfw_window.h"

#include "util/inputcontroller.h"
#include <chrono>

glfw_Window::glfw_Window(const std::string &name, int window_width, int window_height):Window(name,window_width,window_height)
{
}

bool glfw_Window::initWindow()
{



    glfwSetErrorCallback(glfw_Window::error_callback);
    /* Initialize the library */
    if (!glfwInit())
        return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(window_width, window_height, name.c_str(), NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

//    //vsync
    glfwSwapInterval(0);

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
    using namespace std::chrono;

    const long long SKIP_TICKS = 1000000 / ticksPerSecond;
    const int MAX_FRAMESKIP = 20;

    long long next_game_tick = getTicksMS();

    int loops;
    float interpolation;
    update(1.0/60.0);

    bool running = true;
    while( running && !glfwWindowShouldClose(window) ) {

        loops = 0;
        while( getTicksMS() > next_game_tick ) {
            if (loops > MAX_FRAMESKIP){
                cout << "<Gameloop> Warning: Update loop is falling behind." << endl;
                break;
            }
            update(1.0/60.0);

            next_game_tick += SKIP_TICKS;

            loops++;
            //cout << "update"<< endl;
        }

        interpolation = ((float)(getTicksMS() + SKIP_TICKS - next_game_tick )
                        )/ (float) (SKIP_TICKS );
     //   cout << "render "<< interpolation <<  endl;

        renderer->render_intern( interpolation );

       glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
}

void glfw_Window::error_callback(int error, const char* description){

    cout<<"glfw error: "<<error<<" "<<description<<endl;
}
