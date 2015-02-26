#include "camera/camera.h"

#define ANG2RAD 3.14159265358979323846/180.0

Camera::Camera(const string &name) : name(name)
{
}


void Camera::setView(const mat4_t &v){
    view=v;
    recalculateMatrices();
    model = glm::inverse(view);
//    recalculatePlanes();

}

void Camera::setView(const vec3_t &eye,const vec3_t &center,const vec3_t &up){
    setView(glm::lookAt(eye,center,up));
}

 void Camera::updateFromModel(){
     view = glm::inverse(model);
     recalculateMatrices();
 }



//-------------------------------

//int Camera::pointInFrustum(const vec3 &p) {

//    for(int i=0; i < 6; i++) {
//        if (planes[i].distance(p) < 0)
//            return OUTSIDE;
//    }
//    return INSIDE;

//}

//int Camera::sphereInFrustum(const Sphere &s){
//    int result = INSIDE;
//    float distance;

//    for(int i=0; i < 6; i++) {
//        distance = planes[i].distance(s.pos);
//        if (distance < -s.r)
//            return OUTSIDE;
//        else if (distance < s.r)
//            result =  INTERSECT;
//    }
//    return(result);
//}



std::ostream& operator<<(std::ostream& os, const Camera& ca){
    os<<ca.name;
//    os<<"Nearplane= ("<<ca.nw*2<<" x "<<ca.nh*2<<") Farplane= ("<<ca.fw*2<<" x "<<ca.fh*2<<")";
    return os;
}



//===================================================================================================


void PerspectiveCamera::setProj(float fovy, float aspect, float zNear, float zFar){
    fovy = degreesToRadians(fovy);
    this->fovy = fovy;
    this->aspect = aspect;
    this->zNear = zNear;
    this->zFar = zFar;


    tang = (float)tan(fovy * 0.5) ;
    nh = zNear * tang;
    nw = nh * aspect;
    fh = zFar  * tang;
    fw = fh * aspect;

    proj = glm::perspective(fovy,aspect,zNear,zFar);
}

std::ostream& operator<<(std::ostream& os, const PerspectiveCamera& ca){
    os<<"Type: Perspective Camera\n";
    os<<"Name='"<<ca.name<<"' Fovy="<<ca.fovy<<" Aspect="<<ca.aspect<<" zNear="<<ca.zNear<<" zFar="<<ca.zFar<<"\n";
    os<<static_cast<const Camera&>(ca);
    return os;
}

//=========================================================================================================================

void OrthographicCamera::setProj( float left, float right,float bottom,float top,float near,  float far){
    this->left = left;
    this->right = right;
    this->bottom = bottom;
    this->top = top;
    this->zNear = near;
    this->zFar = far;

    nh = (top-bottom)/2;
    nw = (right-left)/2;
    fh = (top-bottom)/2;
    fw = (right-left)/2;
    proj = glm::ortho(left,right,bottom,top,near,far);
}



std::ostream& operator<<(std::ostream& os, const OrthographicCamera& ca){
    os<<"Type: Orthographic Camera";
    os<<"Name='"<<ca.name<<"' left="<<ca.left<<" right="<<ca.right<<" bottom="<<ca.bottom<<" top="<<ca.top<<" zNear="<<ca.zNear<<" zFar="<<ca.zFar<<"\n";
    os<<static_cast<const Camera&>(ca);
    return os;
}

