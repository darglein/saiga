#include "camera/camera.h"

#define ANG2RAD 3.14159265358979323846/180.0

Camera::Camera(const string &name) : name(name)
{
}


void Camera::setView(const mat4_t &v){
    view=v;
    recalculateMatrices();
    model = glm::inverse(view);

    this->position = vec3(model[3]);
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

Camera::IntersectionResult Camera::pointInFrustum(const vec3 &p) {

    for(int i=0; i < 6; i++) {
        if (planes[i].distance(p) < 0)
            return OUTSIDE;
    }
    return INSIDE;

}

Camera::IntersectionResult Camera::sphereInFrustum(const Sphere &s){
    IntersectionResult result = INSIDE;
    float distance;

    for(int i=0; i < 6; i++) {
        distance = planes[i].distance(s.pos);
        if (distance >= s.r){
//            cout<<"outside of plane "<<i<<" "<<planes[i]<<endl;
            return OUTSIDE;
        }else if (distance > -s.r)
            result =  INTERSECT;
    }
    return result;
}



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

void PerspectiveCamera::recalculatePlanes()
{
//    cout<<nh<<" "<<nw<<" "<<fh<<" "<<fw<<endl;
    vec3 dir = -vec3(model[2]);
    vec3 up = vec3(model[1]);
    vec3 right = vec3(model[0]);
//    cout<<"dir "<<dir<<" up "<<up<<endl;

    vec3 nearplanepos = position + dir*zNear;
    vec3 farplanepos = position + dir*zFar;
    //near plane
    planes[0].set(nearplanepos,-dir);
    //far plane
    planes[1].set(farplanepos,dir);


    //calcuate 4 corners of near plane
    vec3 tl = nearplanepos + nh * up - nw * right;
    vec3 tr = nearplanepos + nh * up + nw * right;
    vec3 bl = nearplanepos - nh * up - nw * right;
    vec3 br = nearplanepos - nh * up + nw * right;
    planes[2].set(position,tr,tl); //top
    planes[3].set(position,bl,br); //bottom
    planes[4].set(position,tl,bl); //left
    planes[5].set(position,br,tr); //right
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

void OrthographicCamera::recalculatePlanes()
{

}



std::ostream& operator<<(std::ostream& os, const OrthographicCamera& ca){
    os<<"Type: Orthographic Camera";
    os<<"Name='"<<ca.name<<"' left="<<ca.left<<" right="<<ca.right<<" bottom="<<ca.bottom<<" top="<<ca.top<<" zNear="<<ca.zNear<<" zFar="<<ca.zFar<<"\n";
    os<<static_cast<const Camera&>(ca);
    return os;
}

