/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/lighting/directional_light.h"
#include "saiga/geometry/clipping.h"
#include "saiga/geometry/obb.h"
#include "saiga/imgui/imgui.h"

namespace Saiga {

void DirectionalLightShader::checkUniforms(){
    LightShader::checkUniforms();
    location_direction = getUniformLocation("direction");
    location_ambientIntensity = getUniformLocation("ambientIntensity");
    location_ssaoTexture = getUniformLocation("ssaoTex");
    location_depthTexures = getUniformLocation("depthTexures");
    location_viewToLightTransforms = getUniformLocation("viewToLightTransforms");
    location_depthCuts = getUniformLocation("depthCuts");
    location_numCascades = getUniformLocation("numCascades");
    location_cascadeInterpolateRange = getUniformLocation("cascadeInterpolateRange");
}



void DirectionalLightShader::uploadDirection(vec3 &direction){
    Shader::upload(location_direction,direction);
}

void DirectionalLightShader::uploadAmbientIntensity(float i)
{
    Shader::upload(location_ambientIntensity,i);
}


void DirectionalLightShader::uploadNumCascades(int n)
{
    Shader::upload(location_numCascades,n);
}

void DirectionalLightShader::uploadCascadeInterpolateRange(float r)
{
    Shader::upload(location_cascadeInterpolateRange,r);
}

void DirectionalLightShader::uploadSsaoTexture(std::shared_ptr<raw_Texture> texture)
{

    texture->bind(5);
    Shader::upload(location_ssaoTexture,5);
}

void DirectionalLightShader::uploadDepthTextures(std::vector<std::shared_ptr<raw_Texture> > &textures){

//    int i = 7;
    int startTexture = 6;
    std::vector<int> ids;

    for(int i = 0; i < MAX_CASCADES; ++i){
//    for(auto& t : textures){
        if(i < (int)textures.size()){
            textures[i]->bind(i + startTexture);
            ids.push_back(i + startTexture);

        }else{
            ids.push_back(startTexture);
        }
//        i++;

    }
    Shader::upload(location_depthTexures,ids.size(),ids.data());
}

void DirectionalLightShader::uploadViewToLightTransforms(std::vector<mat4> &transforms)
{
    Shader::upload(location_viewToLightTransforms,transforms.size(),transforms.data());
}

void DirectionalLightShader::uploadDepthCuts(std::vector<float> &depthCuts)
{
    Shader::upload(location_depthCuts,depthCuts.size(),depthCuts.data());
}


//==================================


void DirectionalLight::createShadowMap(int w, int h, int numCascades, ShadowQuality quality){
    SAIGA_ASSERT(numCascades > 0 && numCascades <= MAX_CASCADES);
    this->numCascades = numCascades;
    //    Light::createShadowMap(resX,resY);
    shadowmap = std::make_shared<CascadedShadowmap>(w,h,numCascades,quality);
//    shadowmap->createCascaded(w,h,numCascades);
    orthoBoxes.resize(numCascades);


//     depthCutsRelative = std::vector<float>{0,0.5,1.0};
    depthCutsRelative.resize(numCascades + 1);
     depthCuts.resize(numCascades + 1);

     for(int i = 0; i < numCascades; ++i){
         depthCutsRelative[i] = float(i) / numCascades;
     }
     depthCutsRelative.back() = 1.0f;
}


void DirectionalLight::setDirection(const vec3 &dir){
    direction = glm::normalize(dir);

    vec3 d = -direction;
    vec3 right = normalize(cross(vec3(1,1,0),d));
    vec3 up = normalize(cross(d,right));


    glm::mat3 m;
    m[0] = right;
    m[1] = up;
    m[2] = d;

    vec3 cp = vec3(0);

    this->shadowCamera.setPosition( cp );


    this->shadowCamera.rot = glm::quat_cast( m );

    this->shadowCamera.calculateModel();
    this->shadowCamera.updateFromModel();
}


void DirectionalLight::fitShadowToCamera(Camera *cam)
{
#if 0
    vec3 dir = -direction;
    vec3 right = normalize(cross(vec3(1,1,0),dir));
    vec3 up = normalize(cross(dir,right));

    OBB obb;
    obb.setOrientationScale( normalize(right), normalize(up), normalize(dir) );

    obb.fitToPoints(0,cam->vertices,8);
    obb.fitToPoints(1,cam->vertices,8);
    obb.fitToPoints(2,cam->vertices,8);


    vec3 increase(0,0,5.0);

    float xDiff = 2.0f * length(obb.orientationScale[0]) + increase.x;
    float yDiff = 2.0f * length(obb.orientationScale[1]) + increase.y;
    float zDiff = 2.0f * length(obb.orientationScale[2]) + increase.z;

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


    //    vec4 test = this->cam.proj * this->cam.view * vec4(obb.center,1);
    //    cout << "test " << test << endl;
#else
    //other idea use bounding sphere of frustum
    //make sure shadow box aligned to light fits bounding sphere
    //note: camera movement or rotation doesn't change the size of the shadow box anymore
    //translate the box only by texel size increments to remove flickering




    Sphere boundingSphere = cam->boundingSphere;

    for(int i = 0; i < (int)depthCutsRelative.size(); ++i){
        float a = depthCutsRelative[i];
        depthCuts[i] = (1.0f - a) * cam->zNear + (a) * cam->zFar;
    }

    for(int c = 0 ; c < numCascades; ++c){

        AABB& orthoBox = orthoBoxes[c];

        {
            PerspectiveCamera *pc = static_cast<PerspectiveCamera*>(cam);
            //compute bounding sphere for cascade

            //        vec3 d = -vec3(cam->model[2]);
            vec3 right = vec3(cam->model[0]);
            vec3 up = vec3(cam->model[1]);
            vec3 dir = -vec3(cam->model[2]);


                    float zNear = depthCuts[c]     - cascadeInterpolateRange;
                    float zFar =  depthCuts[c + 1] + cascadeInterpolateRange;
            //        float zNear = cam->zNear;
            //        float zFar = cam->zFar;

//            float zNear = 1;
//            float zFar = 10;

//            if(c == 1){
//                zNear = 10;
//                zFar = 50;
//            }

//            cout << "znear/far: " << zNear << " " << zFar << endl;

            float tang = (float)tan(pc->fovy * 0.5) ;

            float fh = zFar  * tang;
            float fw = fh * pc->aspect;

            vec3 nearplanepos = cam->getPosition() + dir*zNear;
            vec3 farplanepos = cam->getPosition() + dir*zFar;
            vec3 v = farplanepos + fh * up - fw * right;


            vec3 sphereMid = (nearplanepos+farplanepos)*0.5f;
            float r = glm::distance(v,sphereMid);

            boundingSphere.r = r;
            boundingSphere.pos = sphereMid;
        }

        vec3 lightPos = this->shadowCamera.getPosition();

        float r = boundingSphere.r;
        r = ceil(r);

        vec3 smsize = vec3(shadowmap->getSize(),128468);

        vec3 texelSize;
        //    texelSize.x = 2.0f * r / shadowmap.w;
        //    texelSize.y = 2.0f * r / shadowmap.h;
        //    texelSize.z = 0.0001f;
        texelSize = 2.0f * r / smsize;

        //project the position of the actual camera to light space
        vec3 p = boundingSphere.pos;
        glm::mat3 v = glm::mat3(this->shadowCamera.view);
        vec3 t = v * p - v * lightPos;
        t.z = -t.z;




        orthoBox.min = t - vec3(r);
        orthoBox.max = t + vec3(r);

#if 1
        {
            //move camera in texel size increments
            orthoBox.min /= texelSize;
            orthoBox.min = floor(orthoBox.min);
            orthoBox.min *= texelSize;

            orthoBox.max /= texelSize;
            orthoBox.max = floor(orthoBox.max);
            orthoBox.max *= texelSize;
        }
#endif

    }

    //    this->cam.setProj(orthoBox);
    //    this->cam.setProj(
    //                orthoMin.x ,orthoMax.x,
    //                orthoMin.y ,orthoMax.y,
    //                orthoMin.z ,orthoMax.z
    //                );


#if 0
    //test if all cam vertices are in the shadow volume
    for(int i = 0 ;i < 8 ; ++i){
        vec3 v = cam->vertices[i];
        vec4 p = this->cam.proj * this->cam.view * vec4(v,1);
        cout << p << endl;
    }
#endif

#endif
}

void DirectionalLight::fitNearPlaneToScene(AABB sceneBB)
{
    //    vec3 orthoMin(cam.left,cam.bottom,cam.zNear);
    //    vec3 orthoMax(cam.right,cam.top,cam.zFar);


    for(auto& orthoBox : orthoBoxes){

        //transform scene AABB to light space
        auto tris = sceneBB.toTriangles();
        std::vector<PolygonType> trisp;
        for(auto t : tris){
            trisp.push_back( Polygon::toPolygon(t) );
        }
        for(auto& p : trisp){
            for(auto &v : p){
                v = vec3(this->shadowCamera.view * vec4(v,1));
            }
        }


        //clip triangles of scene AABB to the 4 side planes of the frustum
        for(auto &p : trisp){
            p = Clipping::clipPolygonAxisAlignedPlane(p,0,orthoBox.min.x,true);
            p = Clipping::clipPolygonAxisAlignedPlane(p,0,orthoBox.max.x,false);

            p = Clipping::clipPolygonAxisAlignedPlane(p,1,orthoBox.min.y,true);
            p = Clipping::clipPolygonAxisAlignedPlane(p,1,orthoBox.max.y,false);
        }

        float maxZ = -12057135;
        float minZ = 0213650235;

        for(auto& p : trisp){
            for(auto &v : p){
                minZ = std::min(minZ,v.z);
                maxZ = std::max(maxZ,v.z);
            }
        }

        std::swap(minZ,maxZ);
        minZ = -minZ;
        maxZ = -maxZ;

        //    cout << "min max Z " << minZ << " " << maxZ << endl;
        //    cout << "ortho min max Z " << orthoMin.z << " " << orthoMax.z << endl;


        orthoBox.min.z = minZ;
        orthoBox.max.z = maxZ;
    }

    //    this->cam.setProj(orthoBox);
    //    this->cam.setProj(
    //                orthoMin.x ,orthoMax.x,
    //                orthoMin.y ,orthoMax.y,
    //                orthoMin.z ,orthoMax.z
    //                );
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
        std::vector<mat4> viewToLight(numCascades);

        for(int i = 0 ; i < numCascades; ++i){
            this->shadowCamera.setProj(orthoBoxes[i]);
            mat4 shadow = biasMatrix * this->shadowCamera.proj * this->shadowCamera.view * cam->model;
            viewToLight[i] = shadow;
        }

        //        shader.uploadDepthBiasMV(shadow);
        shader.uploadViewToLightTransforms(viewToLight);
        shader.uploadDepthCuts(depthCuts);
        //        shader.uploadDepthTexture(shadowmap->getDepthTexture(0));
        shader.uploadDepthTextures(shadowmap->getDepthTextures());
        shader.uploadShadowMapSize(shadowmap->getSize());
        shader.uploadNumCascades(numCascades);
        shader.uploadCascadeInterpolateRange(cascadeInterpolateRange);
    }

}

void DirectionalLight::bindCascade(int n){
    //    shadowmap.bindCubeFace(gCameraDirections[face].CubemapFace);
    this->shadowCamera.setProj(orthoBoxes[n]);
    shadowmap->bindAttachCascade(n);
}


void DirectionalLight::setDepthCutsRelative(const std::vector<float> &value)
{
    SAIGA_ASSERT((int)value.size() == numCascades + 1);
    depthCutsRelative = value;
}


std::vector<float> DirectionalLight::getDepthCutsRelative() const
{
    return depthCutsRelative;
}

bool DirectionalLight::renderShadowmap(DepthFunction f, UniformBuffer &shadowCameraBuffer)
{
    if(shouldCalculateShadowMap()){

        for(int i = 0; i < getNumCascades(); ++i){
//                light->bindShadowMap();
            bindCascade(i);
            shadowCamera.recalculatePlanes();

            CameraDataGLSL cd(&shadowCamera);
            shadowCameraBuffer.updateBuffer(&cd,sizeof(CameraDataGLSL),0);

            f(&shadowCamera);
//                light->unbindShadowMap();
        }
        return true;
    }else{
        return false;
    }

}

void DirectionalLight::renderImGui()
{
    ImGui::Separator();
    ImGui::Text("DirectionalLight");
    Light::renderImGui();
    ImGui::InputFloat("Cascade Interpolate Range",&cascadeInterpolateRange);
    if(ImGui::Direction("Direction",direction)){
        setDirection(direction);
    }
    ImGui::InputFloat("ambientIntensity",&ambientIntensity);
}

}
