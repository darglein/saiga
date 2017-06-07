#pragma once


#include "saiga/rendering/lighting/light.h"
#include "saiga/camera/camera.h"

class SAIGA_GLOBAL DirectionalLightShader : public LightShader{
public:
    GLint location_direction, location_ambientIntensity;
    GLint location_ssaoTexture;

    virtual void checkUniforms();
    void uploadDirection(vec3 &direction);
    void uploadAmbientIntensity(float i);
    void uploadSsaoTexture(std::shared_ptr<raw_Texture> texture);

};

class SAIGA_GLOBAL DirectionalLight :  public Light
{
protected:


    vec3 direction = vec3(0,-1,0);
    float range = 20.0f;
    float ambientIntensity = 0.2f;

public:
    OrthographicCamera cam;
    //    static void createMesh();
    DirectionalLight();
    virtual ~DirectionalLight(){}

    void bindUniforms(DirectionalLightShader& shader, Camera* cam);

    virtual void createShadowMap(int resX, int resY) override;
    void setAmbientIntensity(float ai);
    float getAmbientIntensity(){return ambientIntensity;}

    /**
     * Sets the light direction in world coordinates.
     * Computes the view matrix for the shadow camera.
     */
    void setDirection(const vec3 &dir);

    /**
     * Computes the left/right, bottom/top and near/far planes of the shadow volume so that,
     * it fits the given camera. This should be called every time the camera is translated
     * or rotated to be sure, that all visible objects have shadows.
     */
    void fitShadowToCamera(Camera* cam);

    /**
     * Computes the near plane of the shadow frustum so that all objects in the scene cast shadows.
     * Objects that do not lay in the camera frustum can cast shadows into it. That's why the method above
     * is not enough. This will overwrite the near plane, so it should be called after fitShadowToCamera.
     */
    void fitNearPlaneToScene(aabb sceneBB);
};


