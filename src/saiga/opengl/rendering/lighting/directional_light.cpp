/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/geometry/clipping.h"
#include "saiga/core/geometry/obb.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"

namespace Saiga
{
void DirectionalLightShader::checkUniforms()
{
    LightShader::checkUniforms();
    location_direction               = getUniformLocation("direction");
    location_ambientIntensity        = getUniformLocation("ambientIntensity");
    location_ssaoTexture             = getUniformLocation("ssaoTex");
    location_depthTexures            = getUniformLocation("depthTexures");
    location_viewToLightTransforms   = getUniformLocation("viewToLightTransforms");
    location_depthCuts               = getUniformLocation("depthCuts");
    location_numCascades             = getUniformLocation("numCascades");
    location_cascadeInterpolateRange = getUniformLocation("cascadeInterpolateRange");
}



void DirectionalLightShader::uploadDirection(vec3& direction)
{
    Shader::upload(location_direction, direction);
}

void DirectionalLightShader::uploadAmbientIntensity(float i)
{
    Shader::upload(location_ambientIntensity, i);
}


void DirectionalLightShader::uploadNumCascades(int n)
{
    Shader::upload(location_numCascades, n);
}

void DirectionalLightShader::uploadCascadeInterpolateRange(float r)
{
    Shader::upload(location_cascadeInterpolateRange, r);
}

void DirectionalLightShader::uploadSsaoTexture(std::shared_ptr<TextureBase> texture)
{
    texture->bind(5);
    Shader::upload(location_ssaoTexture, 5);
}

void DirectionalLightShader::uploadDepthTextures(std::vector<std::shared_ptr<TextureBase> >& textures)
{
    //    int i = 7;
    int startTexture = 6;
    std::vector<int> ids;

    for (int i = 0; i < MAX_CASCADES; ++i)
    {
        //    for(auto& t : textures){
        if (i < (int)textures.size())
        {
            textures[i]->bind(i + startTexture);
            ids.push_back(i + startTexture);
        }
        else
        {
            ids.push_back(startTexture);
        }
        //        i++;
    }
    Shader::upload(location_depthTexures, ids.size(), ids.data());
}

void DirectionalLightShader::uploadDepthTextures(std::shared_ptr<ArrayTexture2D> textures)
{
    textures->bind(6);
    Shader::upload(location_depthTexures, 6);
}

void DirectionalLightShader::uploadViewToLightTransforms(AlignedVector<mat4>& transforms)
{
    Shader::upload(location_viewToLightTransforms, transforms.size(), transforms.data());
}

void DirectionalLightShader::uploadDepthCuts(std::vector<float>& depthCuts)
{
    Shader::upload(location_depthCuts, depthCuts.size(), depthCuts.data());
}


//==================================


void DirectionalLight::createShadowMap(int w, int h, int _numCascades, ShadowQuality quality)
{
    SAIGA_ASSERT(_numCascades > 0 && _numCascades <= MAX_CASCADES);
    this->numCascades = _numCascades;
    //    Light::createShadowMap(resX,resY);
    shadowmap = std::make_shared<CascadedShadowmap>(w, h, _numCascades, quality);
    //    shadowmap->createCascaded(w,h,numCascades);
    orthoBoxes.resize(_numCascades);


    depthCutsRelative.resize(_numCascades + 1);
    depthCuts.resize(_numCascades + 1);

    for (int i = 0; i < _numCascades; ++i)
    {
        depthCutsRelative[i] = float(i) / _numCascades;
    }
    depthCutsRelative.back() = 1.0f;
}


void DirectionalLight::setDirection(const vec3& dir)
{
    direction = normalize(dir);

    vec3 d     = -direction;
    vec3 right = normalize(cross(vec3(1, 1, 0), d));
    vec3 up    = normalize(cross(d, right));


    mat3 m;
    m.col(0) = right;
    m.col(1) = up;
    m.col(2) = d;
    //    col(m, 0) = right;
    //    col(m, 1) = up;
    //    col(m, 2) = d;

    vec3 cp = make_vec3(0);

    this->shadowCamera.setPosition(cp);


    this->shadowCamera.rot = quat_cast(m);

    this->shadowCamera.calculateModel();
    this->shadowCamera.updateFromModel();

    //    std::cout << shadowCamera << std::endl;
    //    std::cout << "dir: " << direction.transpose() << std::endl;
    //    std::cout << m << std::endl;
    //    std::cout << shadowCamera.model << std::endl;
    //    std::cout << shadowCamera.proj << std::endl;
}


void DirectionalLight::fitShadowToCamera(Camera* cam)
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

    float xDiff = 2.0f * length(obb.orientationScale[0]) + increase[0];
    float yDiff = 2.0f * length(obb.orientationScale[1]) + increase[1];
    float zDiff = 2.0f * length(obb.orientationScale[2]) + increase[2];

    shadowNearPlane = 0;
    this->cam.setProj(
                -xDiff / 2.0f ,xDiff / 2.0f,
                -yDiff / 2.0f ,yDiff / 2.0f,
                -zDiff / 2.0f ,zDiff / 2.0f
                );

    this->cam.setPosition( obb.center );

    obb.normalize();
    this->cam.rot = quat_cast( obb.orientationScale );

    this->cam.calculateModel();
    this->cam.updateFromModel();


    //    vec4 test = this->cam.proj * this->cam.view * vec4(obb.center,1);
    //    std::cout << "test " << test << std::endl;
#else
    // other idea use bounding sphere of frustum
    // make sure shadow box aligned to light fits bounding sphere
    // note: camera movement or rotation doesn't change the size of the shadow box anymore
    // translate the box only by texel size increments to remove flickering



    Sphere boundingSphere = cam->boundingSphere;

    for (int i = 0; i < (int)depthCutsRelative.size(); ++i)
    {
        float a      = depthCutsRelative[i];
        depthCuts[i] = (1.0f - a) * cam->zNear + (a)*cam->zFar;
    }

    for (int c = 0; c < numCascades; ++c)
    {
        AABB& orthoBox = orthoBoxes[c];

        {
            PerspectiveCamera* pc = static_cast<PerspectiveCamera*>(cam);
            // compute bounding sphere for cascade

            //        vec3 d = -vec3(cam->model[2]);
            vec3 right = make_vec3(cam->model.col(0));
            vec3 up    = make_vec3(cam->model.col(1));
            vec3 dir   = -make_vec3(cam->model.col(2));


            float zNear = depthCuts[c] - cascadeInterpolateRange;
            float zFar  = depthCuts[c + 1] + cascadeInterpolateRange;
            //        float zNear = cam->zNear;
            //        float zFar = cam->zFar;

            //            float zNear = 1;
            //            float zFar = 10;

            //            if(c == 1){
            //                zNear = 10;
            //                zFar = 50;
            //            }

            //            std::cout << "znear/far: " << zNear << " " << zFar << std::endl;

            float tang = (float)tan(pc->fovy * 0.5);

            float fh = zFar * tang;
            float fw = fh * pc->aspect;

            vec3 nearplanepos = cam->getPosition() + dir * zNear;
            vec3 farplanepos  = cam->getPosition() + dir * zFar;
            vec3 v            = farplanepos + fh * up - fw * right;


            vec3 sphereMid = (nearplanepos + farplanepos) * 0.5f;
            float r        = distance(v, sphereMid);

            boundingSphere.r   = r;
            boundingSphere.pos = sphereMid;
        }

        vec3 lightPos = this->shadowCamera.getPosition();

        float r = boundingSphere.r;
        r       = ceil(r);

        vec3 smsize = make_vec3(make_vec2(shadowmap->getSize()), 128468);

        vec3 texelSize;
        //    texelSize[0] = 2.0f * r / shadowmap.w;
        //    texelSize[1] = 2.0f * r / shadowmap.h;
        //    texelSize[2] = 0.0001f;
        texelSize = 2.0f * r / smsize.array();

        // project the position of the actual camera to light space
        vec3 p = boundingSphere.pos;
        mat3 v = make_mat3(this->shadowCamera.view);
        vec3 t = v * p - v * lightPos;
        t[2]   = -t[2];



        orthoBox.min = t - make_vec3(r);
        orthoBox.max = t + make_vec3(r);

#    if 1
        {
            // move camera in texel size increments
            orthoBox.min = (orthoBox.min.array() / texelSize.array());
            orthoBox.min = (orthoBox.min).array().floor();
            orthoBox.min = orthoBox.min.array() * texelSize.array();

            //            orthoBox.max /= texelSize;
            orthoBox.max = (orthoBox.max.array() / texelSize.array());
            orthoBox.max = (orthoBox.max).array().floor();
            orthoBox.max = orthoBox.max.array() * texelSize.array();
        }
#    endif
    }

    //    this->cam.setProj(orthoBox);
    //    this->cam.setProj(
    //                orthoMin[0] ,orthoMax[0],
    //                orthoMin[1] ,orthoMax[1],
    //                orthoMin[2] ,orthoMax[2]
    //                );

#    if 0
    // test if all cam vertices are in the shadow volume
    for (int i = 0; i < 8; ++i)
    {
        vec3 v = cam->vertices[i];
        vec4 p = shadowCamera.proj * shadowCamera.view * make_vec4(v, 1);
        std::cout << p.transpose() << std::endl;
        //        for (int j = 0; j < 3; ++j) SAIGA_ASSERT(p(j) >= -1 && p(j) <= 1);
    }
#    endif

#endif
}

void DirectionalLight::fitNearPlaneToScene(AABB sceneBB)
{
    //    vec3 orthoMin(cam.left,cam.bottom,cam[2]Near);
    //    vec3 orthoMax(cam.right,cam.top,cam[2]Far);


    for (auto& orthoBox : orthoBoxes)
    {
        // transform scene AABB to light space
        auto tris = sceneBB.toTriangles();
        std::vector<PolygonType> trisp;
        for (auto t : tris)
        {
            trisp.push_back(Polygon::toPolygon(t));
        }
        for (auto& p : trisp)
        {
            for (auto& v : p)
            {
                v = make_vec3(this->shadowCamera.view * make_vec4(v, 1));
            }
        }


        // clip triangles of scene AABB to the 4 side planes of the frustum
        for (auto& p : trisp)
        {
            p = Clipping::clipPolygonAxisAlignedPlane(p, 0, orthoBox.min[0], true);
            p = Clipping::clipPolygonAxisAlignedPlane(p, 0, orthoBox.max[0], false);

            p = Clipping::clipPolygonAxisAlignedPlane(p, 1, orthoBox.min[1], true);
            p = Clipping::clipPolygonAxisAlignedPlane(p, 1, orthoBox.max[1], false);
        }

        float maxZ = -12057135;
        float minZ = 0213650235;

        for (auto& p : trisp)
        {
            for (auto& v : p)
            {
                minZ = std::min(minZ, v[2]);
                maxZ = std::max(maxZ, v[2]);
            }
        }

        std::swap(minZ, maxZ);
        minZ = -minZ;
        maxZ = -maxZ;

        //    std::cout << "min max Z " << minZ << " " << maxZ << std::endl;
        //    std::cout << "ortho min max Z " << orthoMin[2] << " " << orthoMax[2] << std::endl;


        orthoBox.min[2] = minZ;
        orthoBox.max[2] = maxZ;
    }

    //    this->cam.setProj(orthoBox);
    //    this->cam.setProj(
    //                orthoMin[0] ,orthoMax[0],
    //                orthoMin[1] ,orthoMax[1],
    //                orthoMin[2] ,orthoMax[2]
    //                );
}

void DirectionalLight::bindUniforms(DirectionalLightShader& shader, Camera* cam)
{
    shader.uploadColorDiffuse(colorDiffuse);
    shader.uploadColorSpecular(colorSpecular);
    shader.uploadAmbientIntensity(ambientIntensity);

    vec3 viewd = -normalize(make_vec3(cam->view * make_vec4(direction, 0)));
    shader.uploadDirection(viewd);

    mat4 ip = inverse(cam->proj);
    shader.uploadInvProj(ip);

    if (this->hasShadows())
    {
        const mat4 biasMatrix =
            make_mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0);

        AlignedVector<mat4> viewToLight(numCascades);

        for (int i = 0; i < numCascades; ++i)
        {
            this->shadowCamera.setProj(orthoBoxes[i]);
            mat4 shadow    = biasMatrix * this->shadowCamera.proj * this->shadowCamera.view * cam->model;
            viewToLight[i] = shadow;
        }

        //        shader.uploadDepthBiasMV(shadow);
        shader.uploadViewToLightTransforms(viewToLight);
        shader.uploadDepthCuts(depthCuts);
        //        shader.uploadDepthTexture(shadowmap->getDepthTexture(0));
        //        shader.uploadDepthTextures(shadowmap->getDepthTextures());
        shader.uploadDepthTextures(shadowmap->getDepthTexture());
        shader.uploadShadowMapSize(shadowmap->getSize());
        shader.uploadNumCascades(numCascades);
        shader.uploadCascadeInterpolateRange(cascadeInterpolateRange);
    }
}



void DirectionalLight::setDepthCutsRelative(const std::vector<float>& value)
{
    SAIGA_ASSERT((int)value.size() == numCascades + 1);
    depthCutsRelative = value;
}


std::vector<float> DirectionalLight::getDepthCutsRelative() const
{
    return depthCutsRelative;
}

void DirectionalLight::bindCascade(int n)
{
    this->shadowCamera.setProj(orthoBoxes[n]);
    shadowmap->bindAttachCascade(n);
}

bool DirectionalLight::renderShadowmap(DepthFunction f, UniformBuffer& shadowCameraBuffer)
{
    if (shouldCalculateShadowMap())
    {
        for (int i = 0; i < getNumCascades(); ++i)
        {
            bindCascade(i);
            shadowCamera.recalculatePlanes();
            CameraDataGLSL cd(&shadowCamera);
            shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
            f(&shadowCamera);
        }
        return true;
    }
    else
    {
        return false;
    }
}

void DirectionalLight::renderImGui()
{
    Light::renderImGui();
    ImGui::InputFloat("ambientIntensity", &ambientIntensity, 0.1, 1);
    ImGui::InputFloat("Cascade Interpolate Range", &cascadeInterpolateRange);
    if (ImGui::Direction("Direction", direction))
    {
        setDirection(direction);
    }
}

}  // namespace Saiga
