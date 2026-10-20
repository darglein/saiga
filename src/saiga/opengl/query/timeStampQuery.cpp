/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/query/timeStampQuery.h"

#include "saiga/core/util/assert.h"
namespace Saiga
{
QueryObject::QueryObject() {}

QueryObject::~QueryObject()
{
    destroy();
}

QueryObject::QueryObject(const QueryObject& other) : QueryObject()
{
    if (other.id)
    {
        create(true);
    }
}

QueryObject& QueryObject::operator=(const QueryObject& other)
{
    // Like the copy constructor: each object owns its own query, the query state is not copied.
    if (this != &other)
    {
        if (other.id)
            create(true);
        else
            destroy();
    }
    return *this;
}

void QueryObject::create(bool initial_record)
{
    if (!id) glGenQueries(1, &id);
    // prevent potential gl erros.
    if (initial_record)
    {
        record();
        waitTimestamp();
    }
}

void QueryObject::destroy()
{
    if (id)
    {
        glDeleteQueries(1, &id);
        id = 0;
    }
}

void QueryObject::record()
{
    SAIGA_ASSERT(id);
    glQueryCounter(id, GL_TIMESTAMP);
}

void QueryObject::begin()
{
    glBeginQuery(GL_TIME_ELAPSED, id);
}

void QueryObject::end()
{
    glEndQuery(GL_TIME_ELAPSED);
}

bool QueryObject::isAvailable()
{
    SAIGA_ASSERT(id);
    GLint res = 0;
    glGetQueryObjectiv(id, GL_QUERY_RESULT_AVAILABLE, &res);
    return res != 0;
}

GLuint64 QueryObject::getTimestamp()
{
    SAIGA_ASSERT(id);
    GLuint64 time = 0;
    glGetQueryObjectui64v(id, GL_QUERY_RESULT, &time);
    return time;
}

GLuint64 QueryObject::waitTimestamp()
{
    SAIGA_ASSERT(id);
    while (!isAvailable())
    {
    }
    return getTimestamp();
}

}  // namespace Saiga
