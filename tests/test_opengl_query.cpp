/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

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
    // Fails with a GL error if the query of copy was deleted with the source
    copy.record();
    copy.waitTimestamp();

    QueryObject empty;
    copy = empty;
}
