#include "libhello/rendering/lighting/point_light.h"
#include "libhello/util/error.h"
void PointLightShader::checkUniforms(){
    LightShader::checkUniforms();
    location_position = getUniformLocation("position");
    location_attenuation = getUniformLocation("attenuation");
    location_shadowPlanes = getUniformLocation("shadowPlanes");
}

void PointLightShader::upload(const vec3 &pos, float r){
    vec4 v(pos,r);
    Shader::upload(location_position,v);
}

void PointLightShader::upload(vec3 &attenuation){
    Shader::upload(location_attenuation,attenuation);
}

void PointLightShader::uploadShadowPlanes(float f , float n){
    Shader::upload(location_shadowPlanes,vec2(f,n));
}


PointLight::PointLight():PointLight(Sphere())
{


}

PointLight::PointLight(const Sphere &sphere):cam("bla"),sphere(sphere){

    //    translateGlobal(sphere.pos);
}

PointLight& PointLight::operator=(const PointLight& light){
    model = light.model;
    color = light.color;
    attenuation = light.attenuation;
    sphere = light.sphere;
    radius = light.radius;
    return *this;
}

void PointLight::setAttenuation(float c, float l , float q){
    attenuation = vec3(c,l,q);
    this->setColor(vec4(1));


    float a = attenuation.z;
    float b = attenuation.y;
    c = attenuation.x-50.f;


    //solve quadric equation
    float x = (-b+sqrt(b*b-4.0f*a*c)) / (2.0f * a);

    //    sphere.r = x;
    //    scale(vec3(sphere.r));
    setRadius(x);
    //    cout<<"light radius "<<sphere.r<<endl;


}

void PointLight::setSimpleAttenuation(float d, float cutoff){
    //    this->setColor(vec4(1));
    //    sphere.r = d;
    //    scale(vec3(sphere.r));

    //solve (1/r^2)*d^2+(2/r)*d+1=a for r
    //    float r = d/(glm::sqrt(1/cutoff)-1);
    //    cout<<r<<endl;
    //     attenuation = vec3(1,2/r,1/(r*r));

    //solve (0.25/r^2)*d^2+(2/r)*d+1=a for r
    //r = (sqrt(a d^2+3 d^2)+2 d)/(2 (a-1))
    float a = (1/cutoff);
    float r = (sqrt(a*d*d+3*d*d)+2*d)/(2*(a-1));
    attenuation = vec3(1,2/r,0.25/(r*r));
    setRadius(d);
    //    float x = d;
    //    float test = 1/(attenuation.x+attenuation.y*x+attenuation.z*x*x);
    //    cout<<"test erg "<<test<<endl;

    //    cout<<"light radius "<<sphere.r<<endl;

}

void PointLight::setLinearAttenuation(float r, float drop)
{
    float i = 1;
    this->setIntensity(i);


    float cutoff = 1-drop;
    float l = (i/cutoff-1)/r;

    setAttenuation(vec3(1,l,0));

    setRadius(r);


//    for(int i=1;i<13;++i){
//        cout<<"Radius "<<i<<" "<<getIntensity()*getAttenuation(i)<<endl;
//    }
}

float PointLight::getAttenuation(float r){
    return 1.0 / (attenuation.x +
                    attenuation.y * r +
                  attenuation.z * r * r);
}


void PointLight::calculateRadius(float cutoff){
    float a = attenuation.z;
    float b = attenuation.y;
    //    float c = attenuation.x-(getIntensity()/cutoff); //absolute
    float c = attenuation.x-(1.0f/cutoff); //relative

    float x;
    if(a==0)
        x=-c/b;
    else
        x = (-b+sqrt(b*b-4.0f*a*c)) / (2.0f * a);
    setRadius(x);
}

vec3 PointLight::getAttenuation() const
{
    return attenuation;
}

void PointLight::setAttenuation(const vec3 &value)
{
    attenuation = value;
}





float PointLight::getRadius() const
{
    return radius;
}


void PointLight::setRadius(float value)
{
    radius = value;
    this->setScale(vec3(radius));
    //     model[0][0] = radius;
    //     model[1][1] = radius;
    //     model[2][2] = radius;
}

void PointLight::bindUniforms(PointLightShader &shader, Camera *cam){

    //    LightMesh::bindUniforms();
    shader.uploadColor(color);
    shader.uploadModel(model);
    shader.upload(sphere.pos,radius);
    shader.upload(attenuation);
    shader.uploadShadowPlanes(this->cam.zFar,this->cam.zNear);

    mat4 ip = glm::inverse(cam->proj);
    shader.uploadInvProj(ip);

    if(this->hasShadows()){
        shader.uploadShadow(1.0f);
        const glm::mat4 biasMatrix(
                    0.5, 0.0, 0.0, 0.0,
                    0.0, 0.5, 0.0, 0.0,
                    0.0, 0.0, 0.5, 0.0,
                    0.5, 0.5, 0.5, 1.0
                    );

        mat4 shadow = biasMatrix*this->cam.proj * this->cam.view * cam->model;
        shader.uploadDepthBiasMV(shadow);

        shader.uploadDepthTexture(shadowmap.depthTexture);
//        cout<<"hasShadows"<<endl;
    }else{

//         glActiveTexture(GL_TEXTURE0);
//         glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
         shader.uploadDepthTexture(dummyTexture);

        shader.uploadShadow(0.0f);
//         cout<<"not hasShadows"<<endl;
    }

    Error::quitWhenError("PointLight::bindUniforms");
}

void PointLight::bindUniformsStencil(MVPShader& shader){
    shader.uploadModel(model);
}



void PointLight::createShadowMap(int resX, int resY) {
//    cout<<"PointLight::createShadowMap"<<endl;

    shadowmap.createCube(resX,resY);
    this->cam.setProj(90.0f,1,0.1f,50.0);
}





struct CameraDirection
{
    GLenum CubemapFace;
    vec3 Target;
    vec3 Up;
};

static const CameraDirection gCameraDirections[] =
{
    { GL_TEXTURE_CUBE_MAP_POSITIVE_X, vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f) },
    { GL_TEXTURE_CUBE_MAP_NEGATIVE_X, vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f) },
    { GL_TEXTURE_CUBE_MAP_POSITIVE_Y, vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f) },
    { GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 0.0f, -1.0f) },
    { GL_TEXTURE_CUBE_MAP_POSITIVE_Z, vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, -1.0f, 0.0f) },
    { GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, -1.0f, 0.0f) }
};


void PointLight::bindFace(int face){
    shadowmap.bindCubeFace(gCameraDirections[face].CubemapFace);
    shadowmap.depthBuffer.check();
}

void PointLight::calculateCamera(int face){
    vec3 pos(this->getPosition());
    vec3 dir(gCameraDirections[face].Target);
    vec3 up(gCameraDirections[face].Up);
    cam.setView(pos,pos+dir,up);
}

bool PointLight::cullLight(Camera *cam)
{
    Sphere s(position,radius);
    this->culled = cam->sphereInFrustum(s)==Camera::OUTSIDE;
    return culled;
}

