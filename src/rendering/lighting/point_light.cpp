#include "saiga/rendering/lighting/point_light.h"
#include "saiga/util/error.h"
#include "saiga/util/assert.h"

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

PointLight::PointLight(const Sphere &sphere):sphere(sphere){

    //    translateGlobal(sphere.pos);
}

PointLight& PointLight::operator=(const PointLight& light){
    model = light.model;
    colorDiffuse = light.colorDiffuse;
    colorSpecular = light.colorSpecular;
    attenuation = light.attenuation;
    sphere = light.sphere;
    radius = light.radius;
    return *this;
}


void PointLight::setLinearAttenuation(float drop)
{
    float r = 1;
    float cutoff = 1-drop;
    //solve
    // 1/(bx+c) = cutoff, for b
    float c = 1.0f;
    float b = (1.0f/cutoff-c)/r;

    attenuation = vec3(c,b,0);

}


float PointLight::calculateRadius(float cutoff){
    float a = attenuation.z;
    float b = attenuation.y;
    float c = attenuation.x-(1.0f/cutoff); //relative

    float x;
    if(a==0)
        x=-c/b;
    else
        x = (-b+sqrt(b*b-4.0f*a*c)) / (2.0f * a);
    return x;
}

float PointLight::calculateRadiusAbsolute(float cutoff)
{
    float a = attenuation.z;
    float b = attenuation.y;
    float c = attenuation.x-(getIntensity()/cutoff); //absolute

    float x;
    if(a==0)
        x=-c/b;
    else
        x = (-b+sqrt(b*b-4.0f*a*c)) / (2.0f * a);
    return x;
}

vec3 PointLight::getAttenuation() const
{
    return attenuation;
}

float PointLight::getAttenuation(float r){
    float x = r / radius;
    return 1.0 / (attenuation.x +
                    attenuation.y * x +
                  attenuation.z * x * x);
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
}

void PointLight::bindUniforms(std::shared_ptr<PointLightShader> shader, Camera *cam){

    //    LightMesh::bindUniforms();
    shader->uploadColorDiffuse(colorDiffuse);
    shader->uploadColorSpecular(colorSpecular);
    shader->uploadModel(model);
    shader->upload(sphere.pos,radius);
    shader->upload(attenuation);
    shader->uploadShadowPlanes(this->cam.zFar,this->cam.zNear);

    mat4 ip = glm::inverse(cam->proj);
    shader->uploadInvProj(ip);

    if(this->hasShadows()){

        //glm like glsl is column major!
        const mat4 biasMatrix(
                    0.5, 0.0, 0.0, 0.0,
                    0.0, 0.5, 0.0, 0.0,
                    0.0, 0.0, 0.5, 0.0,
                    0.5, 0.5, 0.5, 1.0
                    );

        mat4 shadow = biasMatrix*this->cam.proj * this->cam.view * cam->model;
        shader->uploadDepthBiasMV(shadow);

        shader->uploadDepthTexture(shadowmap.getDepthTexture(0));
        shader->uploadShadowMapSize(shadowmap.getSize());
    }

    assert_no_glerror();
}




void PointLight::createShadowMap(int resX, int resY) {
    shadowmap.createCube(resX,resY);
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
}

void PointLight::calculateCamera(int face){
    vec3 pos(this->getPosition());
    vec3 dir(gCameraDirections[face].Target);
    vec3 up(gCameraDirections[face].Up);
    cam.setView(pos,pos+dir,up);
    cam.setProj(90.0f,1,shadowNearPlane,radius);
}

bool PointLight::cullLight(Camera *cam)
{
    Sphere s(getPosition(),radius);
    this->culled = cam->sphereInFrustum(s)==Camera::OUTSIDE;
//    this->culled = false;
//    cout<<culled<<endl;
    return culled;
}

