#include "saiga/rendering/lighting/directional_light.h"

#include "saiga/geometry/clipping.h"
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

    this->cam.setPosition( cp );


    this->cam.rot = glm::quat_cast( m );

    this->cam.calculateModel();
    this->cam.updateFromModel();
}


void DirectionalLight::setAmbientIntensity(float ai)
{
    ambientIntensity = ai;
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

    vec3 cp = this->cam.getPosition();


    float r = cam->boundingSphere.r;
    r = ceil(r);


    vec3 texelSize;
    texelSize.x = 2.0f * r / shadowmap.w;
    texelSize.y = 2.0f * r / shadowmap.h;
    texelSize.z = 0.0001f;

    //project the position of the actual camera to light space
    vec3 p = cam->boundingSphere.pos;
    glm::mat3 v = glm::mat3(this->cam.view);
    vec3 t = v * p - v * cp;
    t.z = -t.z;


    vec3 orthoMin = t - vec3(r);
    vec3 orthoMax = t + vec3(r);

#if 1
    {
        //move camera in texel size increments
        orthoMin /= texelSize;
        orthoMin = floor(orthoMin);
        orthoMin *= texelSize;

        orthoMax /= texelSize;
        orthoMax = floor(orthoMax);
        orthoMax *= texelSize;
    }
#endif


    this->cam.setProj(
                orthoMin.x ,orthoMax.x,
                orthoMin.y ,orthoMax.y,
                orthoMin.z ,orthoMax.z
                );


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

void DirectionalLight::fitNearPlaneToScene(aabb sceneBB)
{
    vec3 orthoMin(cam.left,cam.bottom,cam.zNear);
    vec3 orthoMax(cam.right,cam.top,cam.zFar);


    //transform scene aabb to light space
    auto tris = sceneBB.toTriangles();
    std::vector<PolygonType> trisp;
    for(auto t : tris){
        trisp.push_back( Polygon::toPolygon(t) );
    }
    for(auto& p : trisp){
        for(auto &v : p){
            v = vec3(this->cam.view * vec4(v,1));
        }
    }

    //clip triangles of scene aabb to the 4 side planes of the frustum

    for(auto &p : trisp){
        p = Clipping::clipPolygonAxisAlignedPlane(p,0,orthoMin.x,true);
        p = Clipping::clipPolygonAxisAlignedPlane(p,0,orthoMax.x,false);

        p = Clipping::clipPolygonAxisAlignedPlane(p,1,orthoMin.y,true);
        p = Clipping::clipPolygonAxisAlignedPlane(p,1,orthoMax.y,false);
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


    orthoMin.z = minZ;
    orthoMax.z = maxZ;

    this->cam.setProj(
                orthoMin.x ,orthoMax.x,
                orthoMin.y ,orthoMax.y,
                orthoMin.z ,orthoMax.z
                );
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


