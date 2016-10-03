
#include "saiga/sdl/sdl_window.h"
#include "saiga/rendering/deferred_renderer.h"

SDLWindow::SDLWindow(WindowParameters windowParameters):Window(windowParameters)
{
}

bool SDLWindow::initWindow()
{
    //Initialization flag
    bool success = true;

    //Initialize SDL
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 ){
        std::cout << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
        return false;
    }

    SDL_DisplayMode current;
    SDL_GetCurrentDisplayMode(0 , &current);


    if(windowParameters.fullscreen()){
        windowParameters.width = current.w;
        windowParameters.height = current.h;
    }

    SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 3 );
    SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 2 );

    if(windowParameters.coreContext)
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE );

    if(windowParameters.debugContext)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);


    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, true);

    /*  \param flags The flags for the window, a mask of any of the following:
    *               ::SDL_WINDOW_FULLSCREEN,    ::SDL_WINDOW_OPENGL,
    *               ::SDL_WINDOW_HIDDEN,        ::SDL_WINDOW_BORDERLESS,
    *               ::SDL_WINDOW_RESIZABLE,     ::SDL_WINDOW_MAXIMIZED,
    *               ::SDL_WINDOW_MINIMIZED,     ::SDL_WINDOW_INPUT_GRABBED,
    *               ::SDL_WINDOW_ALLOW_HIGHDPI.
    */
    Uint32 flags =  SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;

    if(windowParameters.resizeAble) flags |=  SDL_WINDOW_RESIZABLE;
    if(windowParameters.borderLess()) flags |=  SDL_WINDOW_BORDERLESS;
    if(windowParameters.fullscreen()) flags |=  SDL_WINDOW_FULLSCREEN;
    if(windowParameters.resizeAble) flags |=  SDL_WINDOW_RESIZABLE;

    //Create window
    gWindow = SDL_CreateWindow(getName().c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, getWidth(), getHeight(), flags );
    if( gWindow == NULL ){
        std::cout << "Window could not be created! SDL Error: " << SDL_GetError() << std::endl;
        return false;
    }

    //Create context
    gContext = SDL_GL_CreateContext( gWindow );
    if( gContext == NULL ){
        std::cout << "OpenGL context could not be created! SDL Error: " << SDL_GetError() << std::endl;
        return false;
    }

    //Use Vsync
    if( SDL_GL_SetSwapInterval( windowParameters.vsync ? 1 : 0) < 0 ){
        std::cout << "Warning: Unable to set VSync! SDL Error: " << SDL_GetError() << std::endl;
    }


    return success;
}

bool SDLWindow::initInput(){
    //Enable text input
    SDL_StartTextInput();
    return true;
}

bool SDLWindow::shouldClose()
{
    return eventHandler.shouldQuit() || !running;
}

void SDLWindow::checkEvents()
{
    eventHandler.update();
}

void SDLWindow::swapBuffers()
{

    SDL_GL_SwapWindow( gWindow );
}


void SDLWindow::freeContext()
{

    //Disable text input
    SDL_StopTextInput();

    //Destroy window
    SDL_DestroyWindow( gWindow );
    gWindow = NULL;

    //Quit SDL subsystems
    SDL_Quit();
}




