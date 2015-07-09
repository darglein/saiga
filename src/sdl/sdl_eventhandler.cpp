#include "libhello/sdl/sdl_eventhandler.h"

void SDL_EventHandler::update(){
    //Handle events on queue
    SDL_Event e;
    while( SDL_PollEvent( &e ) != 0 )
    {
        //User requests quit
        if( e.type == SDL_QUIT )
        {
            quit = true;
        }
        if (e.type == SDL_KEYDOWN)
        {
            keyPressed(e);
        }
        if (e.type == SDL_KEYUP)
        {
            keyReleased(e);
        }
        /* If the mouse is moving */
         if (e.type == SDL_MOUSEMOTION)
         {
             mouseMoved(e.motion.xrel,e.motion.yrel);
         }
         if (e.type == SDL_MOUSEBUTTONDOWN)
          {
             int key = e.button.button;
            mousePressed(key,e.button.x,e.button.y);
         }
         if (e.type == SDL_MOUSEBUTTONUP)
          {
             int key = e.button.button;
            mouseReleased(key,e.button.x,e.button.y);
         }
    }
}

void SDL_EventHandler::keyPressed(const SDL_Event &e){
//    cout<<"Key pressed: "<<key<<endl;

    for(SDL_KeyListener* listener : keyListener){
        listener->keyPressed(e);
    }
}

void SDL_EventHandler::keyReleased(const SDL_Event &e){
//    cout<<"Key released: "<<key<<endl;
    for(SDL_KeyListener* listener : keyListener){
        listener->keyReleased(e);
    }
}

void SDL_EventHandler::mouseMoved(int relx, int rely){
    for(SDL_MouseListener* listener : mouseListener){
        listener->mouseMoved(relx,rely);
    }
}

void SDL_EventHandler::mousePressed(int key, int x, int y){
//    cout<<"mouse pressed "<<x<<" "<<y<<endl;
    for(SDL_MouseListener* listener : mouseListener){
        listener->mousePressed(key,x,y);
    }
}

void SDL_EventHandler::mouseReleased(int key, int x, int y){
//    cout<<"mouse released "<<x<<" "<<y<<endl;
    for(SDL_MouseListener* listener : mouseListener){
        listener->mouseReleased(key,x,y);
    }
}
