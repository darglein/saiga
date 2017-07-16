#include "saiga/opengl/query/timeStampQuery.h"

namespace Saiga {

TimeStampQuery::TimeStampQuery()
{
}

TimeStampQuery::~TimeStampQuery()
{
    if(id)
        glDeleteQueries(1, &id);
}

void TimeStampQuery::create()
{
    if(!id)
        glGenQueries(1, &id);
    //prevent potential gl erros.
    record();
	waitTimestamp();
}

void TimeStampQuery::record()
{
    glQueryCounter(id, GL_TIMESTAMP);
}

bool TimeStampQuery::isAvailable()
{
    GLint res = 0;
    glGetQueryObjectiv(id,GL_QUERY_RESULT_AVAILABLE,&res);
    return res!=0;
}

GLuint64 TimeStampQuery::getTimestamp()
{
    GLuint64 time = 0;
    glGetQueryObjectui64v(id, GL_QUERY_RESULT, &time);
    return time;
}

GLuint64 TimeStampQuery::waitTimestamp()
{
    while(!isAvailable()){

    }
    return getTimestamp();
}

}
