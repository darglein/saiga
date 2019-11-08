#include "bulletDebugDrawer.h"

#include "saiga/opengl/shader/shaderLoader.h"


namespace Saiga
{
GLDebugDrawer::GLDebugDrawer()
{
    //    setDebugMode(DBG_MAX_DEBUG_DRAW_MODE);
    setDebugMode(DBG_DrawWireframe | DBG_DrawConstraints);

    lineShader = shaderLoader.load<MVPShader>("geometry/deferred_mvp_model_forward.glsl");
}


void GLDebugDrawer::render(btDynamicsWorld* world, Camera* cam)
{
    vertices.clear();
    world->debugDrawWorld();

    if (vertices.size() == 0) return;

    lines.set(vertices, GL_STATIC_DRAW);
    lines.setDrawMode(GL_LINES);

    //    glEnable(GL_POLYGON_OFFSET_LINE);
    //    glPolygonOffset(-5,-5);
    lineShader->bind();
    lineShader->uploadModel(mat4::Identity());
    lines.bindAndDraw();
    lineShader->unbind();
    //    glDisable(GL_POLYGON_OFFSET_LINE);
}

GLDebugDrawer::~GLDebugDrawer() {}

void GLDebugDrawer::drawLine(const btVector3& from, const btVector3& to, const btVector3& fromColor,
                             const btVector3& toColor)
{
    std::cout << "draw line " << std::endl;
}

void GLDebugDrawer::drawLine(const btVector3& from, const btVector3& to, const btVector3& color)
{
    VertexNC v(vec3(from.x(), from.y(), from.z()), vec3(0, 1, 0), vec3(color.x(), color.y(), color.z()));

    vertices.push_back(v);
    v.position = vec4(to.x(), to.y(), to.z(), 1);
    vertices.push_back(v);
}

void GLDebugDrawer::drawSphere(const btVector3& p, btScalar radius, const btVector3& color)
{
    std::cout << "draw shpere" << std::endl;
}

void GLDebugDrawer::drawTriangle(const btVector3& a, const btVector3& b, const btVector3& c, const btVector3& color,
                                 btScalar alpha)
{
    std::cout << "draw triangle" << std::endl;
}

void GLDebugDrawer::drawContactPoint(const btVector3& PointOnB, const btVector3& normalOnB, btScalar distance,
                                     int lifeTime, const btVector3& color)
{
    std::cout << "draw contact point" << std::endl;
}

void GLDebugDrawer::reportErrorWarning(const char* warningString)
{
    std::cout << "GLDebugDrawer error/warning: " << warningString << std::endl;
}

void GLDebugDrawer::draw3dText(const btVector3& location, const char* textString)
{
    std::cout << "draw 3d text" << std::endl;
}

void GLDebugDrawer::setDebugMode(int debugMode)
{
    m_debugMode = debugMode;
}

}  // namespace Saiga
