# Builds a compile-fail target and passes only if the build fails AND its output contains the expected diagnostic
# (PASS_REGULAR_EXPRESSION alone would also pass a successful build that printed the phrase in a warning).
#
#   cmake -DBUILD_DIR=<dir> -DTARGET=<target> -DCONFIG=<config> -DEXPECT=<regex> -P expect_build_failure.cmake

foreach (VAR BUILD_DIR TARGET EXPECT)
    if (NOT DEFINED ${VAR})
        message(FATAL_ERROR "expect_build_failure.cmake: ${VAR} is not set")
    endif ()
endforeach ()

set(BUILD_COMMAND ${CMAKE_COMMAND} --build ${BUILD_DIR} --target ${TARGET})
if (CONFIG)
    list(APPEND BUILD_COMMAND --config ${CONFIG})
endif ()

execute_process(COMMAND ${BUILD_COMMAND} RESULT_VARIABLE RESULT OUTPUT_VARIABLE OUTPUT ERROR_VARIABLE OUTPUT)

if (RESULT EQUAL 0)
    message(FATAL_ERROR "${TARGET} compiled, but it must fail with: ${EXPECT}\n${OUTPUT}")
endif ()
if (NOT OUTPUT MATCHES "${EXPECT}")
    message(FATAL_ERROR "${TARGET} failed, but not with the expected diagnostic: ${EXPECT}\n${OUTPUT}")
endif ()
message(STATUS "${TARGET} failed as expected: ${EXPECT}")
