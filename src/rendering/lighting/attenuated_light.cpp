/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/lighting/attenuated_light.h"
#include "saiga/util/error.h"
#include "saiga/util/assert.h"

namespace Saiga {


void AttenuatedLightShader::checkUniforms(){
    LightShader::checkUniforms();
    location_attenuation = getUniformLocation("attenuation");
}


void AttenuatedLightShader::uploadA(vec3 &attenuation, float cutoffRadius){
    Shader::upload(location_attenuation,vec4(attenuation,cutoffRadius));
}

AttenuatedLight::AttenuatedLight()
{


}

AttenuatedLight& AttenuatedLight::operator=(const AttenuatedLight& light){
    model = light.model;
    colorDiffuse = light.colorDiffuse;
    colorSpecular = light.colorSpecular;
    attenuation = light.attenuation;
    cutoffRadius = light.cutoffRadius;
    return *this;
}


void AttenuatedLight::setLinearAttenuation(float drop)
{
    float r = 1;
    float cutoff = 1-drop;
    //solve
    // 1/(bx+c) = cutoff, for b
    float c = 1.0f;
    float b = (1.0f/cutoff-c)/r;

    attenuation = vec3(c,b,0);

}


float AttenuatedLight::calculateRadius(float cutoff){
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

float AttenuatedLight::calculateRadiusAbsolute(float cutoff)
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

vec3 AttenuatedLight::getAttenuation() const
{
    return attenuation;
}

float AttenuatedLight::getAttenuation(float r){
    float x = r / cutoffRadius;
    return 1.0 / (attenuation.x +
                    attenuation.y * x +
                  attenuation.z * x * x);
}

void AttenuatedLight::setAttenuation(const vec3 &value)
{
    attenuation = value;
}


float AttenuatedLight::getRadius() const
{
    return cutoffRadius;
}


void AttenuatedLight::setRadius(float value)
{
    cutoffRadius = value;
    this->setScale(vec3(cutoffRadius));
}

void AttenuatedLight::bindUniforms(std::shared_ptr<AttenuatedLightShader> shader, Camera *cam){
    shader->uploadColorDiffuse(colorDiffuse);
    shader->uploadColorSpecular(colorSpecular);
    shader->uploadModel(model);
    shader->uploadA(attenuation,cutoffRadius);
    assert_no_glerror();
}



}
