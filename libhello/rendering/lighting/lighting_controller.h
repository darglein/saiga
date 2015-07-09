#pragma once

#include "libhello/glfw/glfw_eventhandler.h"
#include "libhello/geometry/ray.h"
#include "libhello/util/observer.h"

class DeferredLighting;
class Light;

class SAIGA_GLOBAL LightingController : public Subject , public glfw_KeyListener , public glfw_MouseListener{
private:
    DeferredLighting& lighting;

    mat4 oldModel;
    Light* selectedLight = nullptr;

    bool active = true;
    vec2 mouse;
    enum State{
        waiting,
        selected,
        moving,
        rotating
    };
    enum Axis{
        X,Y,Z,UNDEFINED
    };

    State state;
    Axis axis;
public:

    LightingController(DeferredLighting& lighting);
    virtual ~LightingController(){}

    //glfw events
    bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods);
    bool character_event(GLFWwindow* window, unsigned int codepoint);
    bool cursor_position_event(GLFWwindow* window, double xpos, double ypos);
    bool mouse_button_event(GLFWwindow* window, int button, int action, int mods);
    bool scroll_event(GLFWwindow* window, double xoffset, double yoffset);


    void selectPointlight(unsigned int id);
    void selectSpotlight(unsigned int id);
    void selectDirectionallight(unsigned int id);

    Light *getSelectedLight() const;
    void setSelectedLight(Light *value);
    void submitChange();
    void submitNotifyChange();
    void undoChange();
    bool isActive() const;
    void setActive(bool value);
    void duplicateSelected();
private:
    Ray createPixelRay(const vec2 &pixel);
    Light* closestPointLight(const Ray &r);
};


