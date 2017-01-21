#include "saiga/rendering/lighting/lighting_controller.h"

#ifdef USE_GLFW
#include "saiga/util/inputcontroller.h"

#include <saiga/opengl/opengl.h>
#include <GLFW/glfw3.h>

#include "saiga/rendering/lighting/directional_light.h"
#include "saiga/rendering/lighting/point_light.h"
#include "saiga/rendering/lighting/spot_light.h"
#include "saiga/rendering/lighting/deferred_lighting.h"

LightingController::LightingController(DeferredLighting& lighting):lighting(lighting){
    glfw_EventHandler::addKeyListener(this,5);
    glfw_EventHandler::addMouseListener(this,5);

    IC.add("select_pointlight", [this](ICPARAMS){
        unsigned int id = args.next<unsigned int>();
        if(!args.isValid())
            return;
        this->selectPointlight(id);
    });

    IC.add("select_spotlight", [this](ICPARAMS){
        unsigned int id = args.next<unsigned int>();
        if(!args.isValid())
            return;
        this->selectSpotlight(id);
    });

    IC.add("select_directionallight", [this](ICPARAMS){
        unsigned int id = args.next<unsigned int>();
        if(!args.isValid())
            return;
        this->selectDirectionallight(id);
    });

    IC.add("lighting_controller_add_pointlight", [this](ICPARAMS){
        (void)args;
        auto light = this->lighting.createPointLight();
        light->translateGlobal(vec3(3,10,0));
        light->setColorDiffuse(vec4(156./256, 42./256, 0,4));
    });

    IC.add("lighting_controller_toggle", [this](ICPARAMS){
        (void)args;
        this->setActive(!active);
        this->notify();
    });

    IC.add("disable_shadows", [this](ICPARAMS){
        (void)args;
        Light* l = this->getSelectedLight();
        if(l)
            l->disableShadows();
    });

    IC.add("enable_shadows", [this](ICPARAMS){
        (void)args;
        Light* l = this->getSelectedLight();
        if(l)
            l->enableShadows();
    });

    IC.add("create_shadowmap", [this](ICPARAMS){
        unsigned int x = args.next<unsigned int>();
        unsigned int y = args.next<unsigned int>();
        if(!args.isValid())
            return;
        Light* l = this->getSelectedLight();
        if(l)
            l->createShadowMap(x,y);
    });

    setActive(false);
}

bool LightingController::isActive() const
{
    return active;
}

void LightingController::setActive(bool value)
{
    if(value==false){
//        setSelectedLight(nullptr);
    }
    lighting.drawDebug = value;
    active = value;
}

void LightingController::selectPointlight(unsigned int id){
//    if(lighting.pointLights.size()>id){
//        setSelectedLight(lighting.pointLights[id]);
//    }
}

void LightingController::selectSpotlight(unsigned int id){
//    if(lighting.spotLights.size()>id){
//        setSelectedLight(lighting.spotLights[id]);
//    }
}

void LightingController::selectDirectionallight(unsigned int id){
//    if(lighting.directionalLights.size()>id){
//        setSelectedLight(lighting.directionalLights[id]);
//    }
}


Light *LightingController::getSelectedLight() const
{
    return selectedLight;
}

void LightingController::setSelectedLight(Light *value)
{
//    if(selectedLight!=nullptr){
//        selectedLight->setSelected(false);
//        selectedLight->model = oldModel;
//    }
//    selectedLight = value;

//    if(selectedLight!=nullptr){
//        selectedLight->setSelected(true);
//        state = State::selected;
//        oldModel = selectedLight->model;
//    }else{
//        state = State::waiting;
//    }
//    notify(); //notify observers
}

void LightingController::submitChange(){
    if(selectedLight!=nullptr){
        state = State::selected;
        oldModel = selectedLight->model;
    }
}

void LightingController::submitNotifyChange(){
    if(selectedLight!=nullptr){
        state = State::selected;
        oldModel = selectedLight->model;
        notify(); //notify observers
    }
}

void LightingController::undoChange(){
    if(selectedLight!=nullptr){
//        state = State::selected;
        selectedLight->model = oldModel;
    }
}

void LightingController::duplicateSelected(){
//    if(selectedLight!=nullptr){
//        undoChange();
//        if(SpotLight* l = dynamic_cast<SpotLight*>(selectedLight)){
//            (*lighting.createSpotLight()) = *l;
//        }else if(DirectionalLight* l = dynamic_cast<DirectionalLight*>(selectedLight)){
//            (*lighting.createDirectionalLight()) = *l;
//        }else if(PointLight* l = dynamic_cast<PointLight*>(selectedLight)){
//            (*lighting.createPointLight()) = *l;
//        }
//    }
}

bool LightingController::key_event(GLFWwindow* window, int key, int scancode, int action, int mods){
    (void)window;(void)scancode;
    if(!active || action!=GLFW_PRESS)
        return false;


    if(state != State::waiting){
        //wait for key command
        switch(key){
        case GLFW_KEY_G:
            undoChange();
            state=State::moving;
            axis = Axis::UNDEFINED;
            return true;
        case GLFW_KEY_R:
            undoChange();
            state=State::rotating;
            axis = Axis::UNDEFINED;
            return true;
        case GLFW_KEY_D:
            if(mods&GLFW_MOD_CONTROL){
                duplicateSelected();
                return true;
            }
            return false;

        }
    }

    if(state == State::moving || state == State::rotating){
        switch(key){
        case GLFW_KEY_X:
            undoChange();
            axis = Axis::X;
            return true;
        case GLFW_KEY_Y:
            undoChange();
            axis = Axis::Y;
            return true;
        case GLFW_KEY_Z:
            undoChange();
            axis = Axis::Z;
            return true;
        }
    }

    return false;
}

bool LightingController::character_event(GLFWwindow* window, unsigned int codepoint){
    (void)window;(void)codepoint;
    return false;
}

bool LightingController::cursor_position_event(GLFWwindow* window, double xpos, double ypos){
    (void)window;
    if(!active || !active )
        return false;
//    vec2 mousenew = vec2(xpos,ypos);
//    vec2 mouserel = (mousenew-mouse)/vec2(lighting.width,lighting.height);
//    mouserel.x *= (float)lighting.width/(float)lighting.height;
//    mouserel.y  = -mouserel.y;
//    mouse = mousenew;
//    if(selectedLight==nullptr)
//        return false;

//    if(mouserel.x==0 && mouserel.y==0)
//        return false;

//    Light* light =selectedLight;
//    //        vec3 eye = vec3(lighting.view[0][3],lighting.view[1][3],lighting.view[2][3]);

//    vec3 eye_pos = vec3(lighting.inview[3]);
//    float d = glm::distance(vec3(light->getPosition()),eye_pos);


//    if(state==State::moving){
//        if(axis==Axis::X){
//            vec3 screenAxis = glm::normalize(vec3(vec2(lighting.view[0]),0));
//            light->translateGlobal(glm::dot(vec3(mouserel.x,mouserel.y,0),screenAxis)*vec3(d,0,0));
//        }else if(axis==Axis::Y){
//            vec3 screenAxis = glm::normalize(vec3(vec2(lighting.view[1]),0));
//            light->translateGlobal(glm::dot(vec3(mouserel.x,mouserel.y,0),screenAxis)*vec3(0,d,0));
//        }else if(axis==Axis::Z){
//            vec3 screenAxis = glm::normalize(vec3(vec2(lighting.view[2]),0));
//            light->translateGlobal(glm::dot(vec3(mouserel.x,mouserel.y,0),screenAxis)*vec3(0,0,d));
//        }else {
//            vec4 trans = vec4(mouserel.x,mouserel.y,0,0)*d;
//            trans = lighting.inview*trans;
//            light->translateGlobal(trans);
//        }
//    }

//    if(state==State::rotating){
//        vec3 a;
//        if(axis==Axis::X){
//            a = vec3(1,0,0);
//        }else if(axis==Axis::Y){
//            a = vec3(0,1,0);
//        }else if(axis==Axis::Z){
//            a = vec3(0,0,1);
//        }else {
//            a = vec3(lighting.inview[2]);
//        }


//        vec4 lp = vec4(light->getPosition(),1);
//        vec4 lpscreen = lighting.proj*lighting.view*lp;
//        lpscreen = lpscreen/lpscreen.w;
//        vec2 mousescreen = mouse/vec2(lighting.width,lighting.height);
//        mousescreen.y = 1.0f-mousescreen.y;
//        mousescreen = mousescreen*2.0f-1.0f;

//        vec2 v1 = glm::normalize(vec2(mousescreen)-vec2(lpscreen));
//        vec2 v2 = glm::normalize(vec2(mousescreen)-mouserel-vec2(lpscreen));


//        float angle =  glm::atan(v2.y,v2.x) - glm::atan(v1.y,v1.x);
////            float angle = glm::acos(glm::dot(v1,v2));

//        //check if axis points towards screen
//        if(glm::dot(vec3(lighting.inview[2]),a)>0){
//            angle = -angle;
//        }
//        light->rotateGlobal(a,glm::degrees(angle));
//    }

//    light->calculateModel();


    return false;
}

bool LightingController::mouse_button_event(GLFWwindow* window, int button, int action, int mods){
    (void)window;(void)mods;
//    if(!active || action!=GLFW_PRESS)
//        return false;

//    if(button==GLFW_MOUSE_BUTTON_1){
//        if(state!=State::waiting)
//            submitNotifyChange();
//    }
//    //select closest light with right click
//    if(button==GLFW_MOUSE_BUTTON_2){
//        Ray r = createPixelRay(mouse);
//        setSelectedLight(closestPointLight(r));
//        return true;
//    }

    return false;
}

//Ray LightingController::createPixelRay(const vec2 &pixel){
//    vec4 p = vec4(2*pixel.x/lighting.width-1.f,-(2*pixel.y/lighting.height-1.f),0,1.f);
//    p = glm::inverse(lighting.proj)*p;
//    p /= p.w;
////    vec4 der = lighting.inview*p;
//    vec4 der = glm::inverse(lighting.view)*p;
//    vec3 origin = vec3(glm::inverse(lighting.view)[3]);
//    return Ray(glm::normalize(vec3(der)-origin),origin);
//}

Light *LightingController::closestPointLight(const Ray &r){
    PointLight* closest = nullptr;
//    float t = 19857985275; //infinite

//    for(PointLight* &obj : lighting.pointLights){
//        Sphere s(vec3(obj->getPosition()),0.05*obj->getRadius());
//        float x1,x2;
//        if(r.intersectSphere(s,x1,x2)){
//            if(x1>0 && x1<t){
//                closest = obj;
//                t = x1;
//            }
//        }
//    }
//    for(SpotLight* &obj : lighting.spotLights){
//        Sphere s(vec3(obj->getPosition()),0.05*obj->getRadius());
//        float x1,x2;
//        if(r.intersectSphere(s,x1,x2)){
//            if(x1>0 && x1<t){
//                closest = obj;
//                t = x1;
//            }
//        }
//    }
    return closest;
}

bool LightingController::scroll_event(GLFWwindow* window, double xoffset, double yoffset){
    (void)window;(void)xoffset;(void)yoffset;
    return false;
}

#endif
