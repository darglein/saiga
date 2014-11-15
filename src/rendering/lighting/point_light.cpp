#include "libhello/rendering/lighting/point_light.h"

void PointLightShader::checkUniforms(){
    LightShader::checkUniforms();
    location_position = getUniformLocation("position");
    location_attenuation = getUniformLocation("attenuation");
}

void PointLightShader::upload(const vec3 &pos, float r){
    vec4 v(pos,r);
    Shader::upload(location_position,v);
}

void PointLightShader::upload(vec3 &attenuation){
    Shader::upload(location_attenuation,attenuation);
}




PointLight::PointLight():PointLight(Sphere())
{
}

PointLight::PointLight(const Sphere &sphere):sphere(sphere){

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

void PointLight::calculateRadius(float cutoff){
    float a = attenuation.z;
    float b = attenuation.y;
//    float c = attenuation.x-(getIntensity()/cutoff); //absolute
    float c = attenuation.x-(1.0f/cutoff); //relative

    float x = (-b+sqrt(b*b-4.0f*a*c)) / (2.0f * a);
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

 void PointLight::bindUniforms(PointLightShader &shader){
 //    LightMesh::bindUniforms();
     shader.uploadColor(color);
     shader.uploadModel(model);
     shader.upload(sphere.pos,sphere.r);
     shader.upload(attenuation);
 }

  void PointLight::bindUniformsStencil(MVPShader& shader){
      shader.uploadModel(model);
  }

// void PointLight::drawNoShaderBind(){
//     bindUniforms();
//     buffer.bindAndDraw();
// }

// void PointLight::drawNoShaderBindStencil(){
//     bindUniformsStencil();
//     buffer.bindAndDraw();
// }

// void PointLight::drawRaw(){
//     buffer.bindAndDraw();
// }
