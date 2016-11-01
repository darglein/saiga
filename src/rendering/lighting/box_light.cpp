#include "saiga/rendering/lighting/box_light.h"


void BoxLightShader::checkUniforms(){
    DirectionalLightShader::checkUniforms();
}





//==================================


BoxLight::BoxLight()
{



}

void BoxLight::createShadowMap(int resX, int resY){
    Light::createShadowMap(resX,resY);
}




void BoxLight::bindUniforms(BoxLightShader &shader, Camera *cam){
    shader.uploadColorDiffuse(colorDiffuse);
    shader.uploadColorSpecular(colorSpecular);
    shader.uploadAmbientIntensity(ambientIntensity);
    shader.uploadModel(model);

//    vec3 viewd = -glm::normalize(vec3((*view)*vec4(direction,0)));
    vec3 viewd = -glm::normalize(vec3(cam->view*model[2]));
    shader.uploadDirection(viewd);

    mat4 ip = glm::inverse(cam->proj);
    shader.uploadInvProj(ip);

    if(this->hasShadows()){
        const mat4 biasMatrix(
                    0.5, 0.0, 0.0, 0.0,
                    0.0, 0.5, 0.0, 0.0,
                    0.0, 0.0, 0.5, 0.0,
                    0.5, 0.5, 0.5, 1.0
                    );

        mat4 shadow = biasMatrix*this->cam.proj * this->cam.view * cam->model;
        shader.uploadDepthBiasMV(shadow);

        shader.uploadDepthTexture(shadowmap.depthTexture);
        shader.uploadShadowMapSize(shadowmap.w,shadowmap.h);
    }

}

void BoxLight::calculateCamera(){
    float length = scale.z;

    vec3 dir = glm::normalize(vec3(this->getDirection()));
    vec3 pos = getPosition() - dir * length * 0.5f;
    vec3 up = vec3(getUpVector());

    cam.setView(pos,pos+dir,up);
    cam.setProj(-scale.x,scale.x,-scale.y,scale.y,shadowNearPlane,scale.z*2.0f);

}

bool BoxLight::cullLight(Camera *cam)
{
    //do an exact frustum-frustum intersection if this light casts shadows, else do only a quick check.
    if(this->hasShadows())
        this->culled = !this->cam.intersectSAT(cam);
    else
        this->culled = cam->sphereInFrustum(this->cam.boundingSphere)==Camera::OUTSIDE;
	//std::cout << "boxlight culled " <<this->culled << std::endl;
    //culled  = false;
    return culled;
}

