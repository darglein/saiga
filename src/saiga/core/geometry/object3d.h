/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

namespace Saiga
{
class SAIGA_CORE_API Object3D
{
   public:
    mat4 model = mat4::Identity();


    // required for non uniform scaled rotations
    // TODO: extra class so uniform objects are faster
    quat rot      = quat::Identity();
    vec4 scale    = make_vec4(1);
    vec4 position = make_vec4(0);


    //    Object3D(){ SAIGA_ASSERT( (size_t)this%16==0); }
    mat4 getModelMatrix() const;
    void getModelMatrix(mat4& model) const;
    void calculateModel();

    vec3 getPosition() const;   // returns global position
    vec4 getDirection() const;  // returns looking direction
    vec4 getRightVector() const;
    vec4 getUpVector() const;

    void setSimpleDirection(const vec3& dir);  // sets looking direction to dir, up to (0,1,0) and right to
                                               // cross(dir,up)

    void translateLocal(const vec3& d);
    void translateGlobal(const vec3& d);
    void translateLocal(const vec4& d);
    void translateGlobal(const vec4& d);

    void rotateLocal(const vec3& axis, float angle);  // rotate around local axis (this is much faster than
                                                      // rotateGlobal)
    void rotateGlobal(vec3 axis, float angle);
    void rotateAroundPoint(const vec3& point, const vec3& axis, float angle);

    vec3 getScale() const;
    void setScale(const vec3& s);
    void multScale(const vec3& s);

    static quat getSimpleDirectionQuat(const vec3& dir);

    // todo: remove virtual methodes
    ~Object3D() {}
    void setPosition(const vec3& cords);
    void setPosition(const vec4& cords);
    void turn(float angleX, float angleY, const vec3& up);
    void turnLocal(float angleX, float angleY);


    void setModelMatrix(const mat4& model);
    void setViewMatrix(const mat4& view);

    // Correct linear spherical interpolation between the two object states.
    static Object3D interpolate(const Object3D& a, const Object3D& b, float alpha);

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const Object3D& ob);
};


inline mat4 Object3D::getModelMatrix() const
{
    return createTRSmatrix(position, rot, scale);
}


inline void Object3D::getModelMatrix(mat4& mod) const
{
    mod = getModelMatrix();
}

inline void Object3D::calculateModel()
{
    model = createTRSmatrix(position, rot, scale);
}

inline vec3 Object3D::getPosition() const
{
    return make_vec3(position);
}

inline vec4 Object3D::getDirection() const
{
    return make_vec4(rot * vec3(0, 0, 1), 0);
}

inline vec4 Object3D::getRightVector() const
{
    //    return rot * vec4(1, 0, 0, 0);
    return make_vec4(rot * vec3(1, 0, 0), 0);
}

inline vec4 Object3D::getUpVector() const
{
    //    return rot * vec4(0, 1, 0, 0);
    return make_vec4(rot * vec3(0, 1, 0), 0);
}

inline void Object3D::setPosition(const vec3& cords)
{
    position = make_vec4(cords, 1);
}

inline void Object3D::setPosition(const vec4& cords)
{
    position = cords;
}

inline void Object3D::translateLocal(const vec4& d)
{
    translateLocal(make_vec3(d));
}

inline void Object3D::translateGlobal(const vec4& d)
{
    translateGlobal(make_vec3(d));
}

inline void Object3D::translateLocal(const vec3& d)
{
    vec4 d2 = make_vec4(rot * d, 1);
    translateGlobal(d2);
}

inline void Object3D::translateGlobal(const vec3& d)
{
    position += make_vec4(d, 0);
}

inline void Object3D::rotateLocal(const vec3& axis, float angle)
{
    this->rot = rotate(this->rot, radians(angle), axis);
}

inline void Object3D::rotateGlobal(vec3 axis, float angle)
{
    axis      = (inverse(rot) * axis);
    axis      = normalize(axis);
    this->rot = rotate(this->rot, radians(angle), axis);
}

inline vec3 Object3D::getScale() const
{
    return make_vec3(scale);
}

inline void Object3D::setScale(const vec3& s)
{
    scale = make_vec4(s, 1);
}

inline void Object3D::multScale(const vec3& s)
{
    setScale(getScale().array() * s.array());
}


inline void Object3D::setViewMatrix(const mat4& view)
{
    setModelMatrix(inverse(view));
}

}  // namespace Saiga
