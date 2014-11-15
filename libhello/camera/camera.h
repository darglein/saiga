#ifndef CAMERA_H
#define CAMERA_H

#include "libhello/util/glm.h"
#include "libhello/rendering/object3d.h"

#include <string>
#include <iostream>
typedef glm::mat4 mat4_t;
typedef glm::vec3 vec3_t;

using std::string;

class Camera : public Object3D{
public:
    string name;

    mat4 view;
    mat4 proj;
    mat4 viewProj;

    float zNear,  zFar;
    float nw,nh,fw,fh; //dimensions of near and far plane

    Camera(const string &name);

    void setView(const mat4 &v);
    void setView(const vec3 &eye,const vec3 &center,const vec3 &up);


    void setProj(const mat4_t &p){proj=p;recalculateMatrices();}
//    virtual void setProj( double fovy, double aspect, double zNear, double zFar){}
//    virtual void setProj( float left, float right,float bottom,float top,float near,  float far){}

    void updateFromModel();
    void recalculateMatrices(){viewProj = proj * view;}
//    virtual void recalculatePlanes() = 0;

//    //culling stuff
//    int pointInFrustum(const vec3_t &p);
//    int sphereInFrustum(const Sphere &s);


//    void draw();
private:
    friend std::ostream& operator<<(std::ostream& os, const Camera& ca);
};

//========================= PerspectiveCamera =========================

class PerspectiveCamera : public Camera{
public:
     double fovy,  aspect;
      float tang;
    PerspectiveCamera(const string &name):Camera(name){}
     void setProj( double fovy, double aspect, double zNear, double zFar);
       friend std::ostream& operator<<(std::ostream& os, const PerspectiveCamera& ca);

};

//========================= OrthographicCamera =========================

class OrthographicCamera : public Camera{
public:
     float left,right,bottom,top;
    OrthographicCamera(const string &name):Camera(name){}
     void setProj( float left, float right,float bottom,float top,float near,  float far);

       friend std::ostream& operator<<(std::ostream& os, const OrthographicCamera& ca);

};

#endif // CAMERA_H
