#if 0
#include <SDL2/SDL.h>
#include <thread>
#include <chrono>
#include <GL/glew.h>
#include <iostream>

static SDL_Window* window;



void swapTestGL()
{
    SDL_Init( SDL_INIT_VIDEO );
    window = SDL_CreateWindow("test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 1280,720, SDL_WINDOW_OPENGL);
    auto gContext = SDL_GL_CreateContext( window );
    SDL_GL_SetSwapInterval( 0 ) ;
    glewInit();


    auto render = [&]()
    {

        static int i = 0;
        glClearColor( (i++)%1000 / 1000.0f,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT );
        SDL_GL_SwapWindow( window );
    };

    for(int i = 0; i < 100; ++i)
        render();


    auto start = std::chrono::high_resolution_clock::now();
    int count =0;
	float time = 3;

    while(true)
    {
        render();
        count ++;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now - start;
        if(duration > std::chrono::seconds( (int)time))
            break;
    }
    std::cout << "Rendered " << count << " frames in 10 seconds. -> " << count/ time << " fps." << std::endl;

    SDL_DestroyWindow( window );
    SDL_Quit();
}
#endif
