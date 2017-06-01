#include "saiga/rendering/lighting/directional_light.h"

#include "saiga/geometry/obb.h"

void DirectionalLightShader::checkUniforms(){
    LightShader::checkUniforms();
    location_direction = getUniformLocation("direction");
    location_ambientIntensity = getUniformLocation("ambientIntensity");
    location_ssaoTexture = getUniformLocation("ssaoTex");
}



void DirectionalLightShader::uploadDirection(vec3 &direction){
    Shader::upload(location_direction,direction);
}

void DirectionalLightShader::uploadAmbientIntensity(float i)
{
    Shader::upload(location_ambientIntensity,i);
}

void DirectionalLightShader::uploadSsaoTexture(std::shared_ptr<raw_Texture> texture)
{

        texture->bind(6);
        Shader::upload(location_ssaoTexture,6);
}


//==================================

//void DirectionalLight::createMesh(){
//    Plane p(vec3(0),vec3(0,1,0));
//    auto* m = TriangleMeshGenerator::createFullScreenQuadMesh();
//    m->createBuffers(buffer);
//}

DirectionalLight::DirectionalLight()
{



}

void DirectionalLight::createShadowMap(int resX, int resY){
    Light::createShadowMap(resX,resY);
    range = 20.0f;
//    cam.setProj(-range,range,-range,range,shadowNearPlane,50.0f);

}


void DirectionalLight::setDirection(const vec3 &dir){
    direction = glm::normalize(dir);


}

void DirectionalLight::setFocus(const vec3 &pos){
//    cam.setView(pos-direction*range, pos, vec3(0,1,0));
}

void DirectionalLight::setAmbientIntensity(float ai)
{
    ambientIntensity = ai;
}

void DirectionalLight::fitShadowToCamera(Camera *cam)
{
    vec3 dir = -direction;
    vec3 right = cross(vec3(0,1,0),dir);
    vec3 up = cross(dir,right);

    OBB obb;
    obb.setOrientationScale( normalize(right), normalize(up), normalize(dir) );

    obb.fitToPoints(0,cam->vertices,8);
    obb.fitToPoints(1,cam->vertices,8);
    obb.fitToPoints(2,cam->vertices,8);




//    obb.orientationScale[0] *= increase[0];
//    obb.orientationScale[0] *= increase[1];
//    obb.orientationScale[0] *= increase[2];

//    float xMin = 234235125, xMax = -34853690;
//    float yMin = 234235125, yMax = -34853690;
//    float zMin = 234235125, zMax = -34853690;


//    //project vertices of camera frustum to direction vector
//    for(int i = 0 ; i < 8 ; ++i){
//        float x = dot(right,cam->vertices[i]);
//        xMin = glm::min(xMin,x); xMax = glm::max(xMax,x);

//        float y = dot(up,cam->vertices[i]);
//        yMin = glm::min(yMin,y); yMax = glm::max(yMax,y);

//        float z = dot(direction,cam->vertices[i]);
//        zMin = glm::min(zMin,z); zMax = glm::max(zMax,z);
//    }
    vec3 increase(0,0,5.0);

    float xDiff = 2.0f * length(obb.orientationScale[0]) + increase.x;
    float yDiff = 2.0f * length(obb.orientationScale[1]) + increase.y;
    float zDiff = 2.0f * length(obb.orientationScale[2]) + increase.z;
//    float yDiff = yMax - yMin;
//    float zDiff = zMax - zMin;
//    cout << xMin << "," << xMax << " "  << yMin << "," << yMax << " "  << zMin << "," << zMax << endl;
//    cout << xDiff << " " << yDiff << " " << zDiff << endl;
//    cout << obb.center << endl << obb.orientationScale << endl << endl;



    //other idea use bounding sphere of frustum
    //make sure shadow box aligned to light fits bounding sphere
    //note: camera movement or rotation doesn't change the size of the shadow box anymore
    //translate the box only by texel size increments to remove flickering


    shadowNearPlane = 0;
    this->cam.setProj(
                -xDiff / 2.0f ,xDiff / 2.0f,
                -yDiff / 2.0f ,yDiff / 2.0f,
                -zDiff / 2.0f ,zDiff / 2.0f
                );

    this->cam.setPosition( obb.center );

    obb.normalize();
    this->cam.rot = glm::quat_cast( obb.orientationScale );

    this->cam.calculateModel();
    this->cam.updateFromModel();
}

void DirectionalLight::bindUniforms(DirectionalLightShader &shader, Camera *cam){
    shader.uploadColorDiffuse(colorDiffuse);
    shader.uploadColorSpecular(colorSpecular);
    shader.uploadAmbientIntensity(ambientIntensity);

    vec3 viewd = -glm::normalize(vec3(cam->view*vec4(direction,0)));
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


