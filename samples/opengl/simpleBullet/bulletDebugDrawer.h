#pragma once


#include "saiga/core/camera/camera.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/vertex.h"
#include "saiga/opengl/vertexBuffer.h"

#include "LinearMath/btIDebugDraw.h"
#include "btBulletDynamicsCommon.h"

namespace Saiga
{
class GLDebugDrawer : public btIDebugDraw
{
   private:
    int m_debugMode;
    std::shared_ptr<MVPShader> lineShader;
    VertexBuffer<VertexNC> lines;
    std::vector<VertexNC> vertices;


   public:
    GLDebugDrawer();
    virtual ~GLDebugDrawer();


    void render(btDynamicsWorld* world, Camera* cam);

    virtual void drawLine(const btVector3& from, const btVector3& to, const btVector3& fromColor,
                          const btVector3& toColor);

    virtual void drawLine(const btVector3& from, const btVector3& to, const btVector3& color);

    virtual void drawSphere(const btVector3& p, btScalar radius, const btVector3& color);

    virtual void drawTriangle(const btVector3& a, const btVector3& b, const btVector3& c, const btVector3& color,
                              btScalar alpha);

    virtual void drawContactPoint(const btVector3& PointOnB, const btVector3& normalOnB, btScalar distance,
                                  int lifeTime, const btVector3& color);

    virtual void reportErrorWarning(const char* warningString);

    virtual void draw3dText(const btVector3& location, const char* textString);

    virtual void setDebugMode(int debugMode);

    virtual int getDebugMode() const { return m_debugMode; }
};

}  // namespace Saiga
