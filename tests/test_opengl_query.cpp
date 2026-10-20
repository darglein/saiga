/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/query/timeStampQuery.h"

#include "opengl_test_context.h"

using namespace Saiga;

class OpenGLQuery : public OpenGLTest
{
};

TEST_F(OpenGLQuery, Timestamp)
{
    QueryObject query;
    query.create(false);
    query.record();
    EXPECT_GT(query.waitTimestamp(), 0u);
}

// The copy constructor created a new query, but the implicit copy assignment shared the query of the source,
// which was then deleted twice.
TEST_F(OpenGLQuery, CopyAssignment)
{
    QueryObject copy;
    {
        QueryObject query;
        query.create(true);
        copy = query;
    }
    // Recording on a query that was deleted with the source is a GL error. Check it before waiting, which would
    // otherwise never end.
    copy.record();
    ASSERT_EQ(glGetError(), GL_NO_ERROR);
    copy.waitTimestamp();

    QueryObject empty;
    copy = empty;
}

// The copy assignment kept the query of the destination. If it had been used with begin()/end(), it is a
// GL_TIME_ELAPSED query, and recording a timestamp on it is a GL error.
TEST_F(OpenGLQuery, CopyAssignmentToElapsedQuery)
{
    QueryObject elapsed;
    elapsed.create(false);
    elapsed.begin();
    elapsed.end();
    elapsed.waitTimestamp();

    QueryObject query;
    query.create(true);
    elapsed = query;
    ASSERT_EQ(glGetError(), GL_NO_ERROR);

    elapsed.record();
    ASSERT_EQ(glGetError(), GL_NO_ERROR);
    EXPECT_GT(elapsed.waitTimestamp(), 0u);
}

TEST_F(OpenGLQuery, OpenGLTimer)
{
    OpenGLTimer timer;
    timer.start();
    glFinish();
    timer.stop();
    EXPECT_GE(timer.getTimeMS(), 0.f);
}

TEST_F(OpenGLQuery, MultiFrameTimerTimestamps)
{
    MultiFrameOpenGLTimer timer(true);
    timer.create();
    for (int i = 0; i < 4; ++i)
    {
        timer.Start();
        timer.Stop();
        ASSERT_EQ(glGetError(), GL_NO_ERROR) << i;
        // The time of the previous frame: begin and end of the same frame
        EXPECT_LT(timer.getTimeMS(), 1000.f) << i;
    }
}

// With GL_TIME_ELAPSED queries, the first Stop() read the result of a query that had never been started, which is a
// GL_INVALID_OPERATION.
TEST_F(OpenGLQuery, MultiFrameTimerElapsed)
{
    MultiFrameOpenGLTimer timer(false);
    timer.create();
    for (int i = 0; i < 4; ++i)
    {
        timer.Start();
        timer.Stop();
        ASSERT_EQ(glGetError(), GL_NO_ERROR) << i;
        EXPECT_LT(timer.getTimeMS(), 1000.f) << i;
    }
}
