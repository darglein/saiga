############# Check which compiler is used ############

set(SAIGA_CXX_FLAGS_GNU OFF)
set(SAIGA_CXX_FLAGS_MSVC OFF)

set(SAIGA_COMPILER_STRING ${CMAKE_CXX_COMPILER_ID})

if ("${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" MATCHES "GNU")
    set(SAIGA_CXX_FLAGS_GNU ON)
elseif ("${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" MATCHES "MSVC")
    set(SAIGA_CXX_FLAGS_MSVC ON)
else ()
    message(FATAL_ERROR "Unknown CXX Compiler frontend. '${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}'")
endif ()


message(STATUS "CMAKE_CXX_COMPILER_ID:   ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_SIMULATE_ID:   ${CMAKE_CXX_SIMULATE_ID}")
message(STATUS "CMAKE_CXX_COMPILER_FRONTEND_VARIANT:   ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
message(STATUS "Detected Compiler:  ${SAIGA_COMPILER_STRING}")
message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "SAIGA_CXX_FLAGS_GNU: ${SAIGA_CXX_FLAGS_GNU}")
message(STATUS "SAIGA_CXX_FLAGS_MSVC: ${SAIGA_CXX_FLAGS_MSVC}")


######### warnings #########


if (SAIGA_CXX_FLAGS_MSVC)
    # Force to always compile with W1
    if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else ()
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
    endif ()
endif ()

if (SAIGA_CXX_FLAGS_GNU)
    list(APPEND SAIGA_CXX_FLAGS "-Wall")
    list(APPEND SAIGA_CXX_FLAGS "-Werror=return-type")
    list(APPEND SAIGA_CXX_FLAGS "-Wno-strict-aliasing")
    list(APPEND SAIGA_CXX_FLAGS "-Wno-sign-compare")
    list(APPEND SAIGA_CXX_FLAGS "-Wno-missing-braces")
    list(APPEND SAIGA_PRIVATE_CXX_FLAGS "-fvisibility=hidden")
endif ()

#if (SAIGA_CXX_WCLANG)
# Fixes: cannot use 'throw' with exceptions disabled
#list(APPEND SAIGA_CXX_FLAGS "-Xclang -fcxx-exceptions")
# some eigen header generates this warning
#    add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
#endif ()


if (SAIGA_CXX_FLAGS_MSVC)

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
        #multiprocessor compilation for visual studio
        list(APPEND SAIGA_CXX_FLAGS "/MP")
    endif ()

    # required for some crazy eigen stuff
    list(APPEND SAIGA_CXX_FLAGS "/bigobj")

    # non dll-interface struct 'xx' used as base for dll-interface struct 'xx'
    list(APPEND SAIGA_CXX_FLAGS "/wd4275")


    #    set(CMAKE_CXX_FLAGS "/MP ${CMAKE_CXX_FLAGS}")
    add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif ()

######### floating point #########

if (SAIGA_STRICT_FP)
    if (SAIGA_CXX_FLAGS_GNU)
        list(APPEND SAIGA_CXX_FLAGS "-msse2" "-mfpmath=sse")
    endif ()
    if (SAIGA_CXX_FLAGS_MSVC)
        list(APPEND SAIGA_CXX_FLAGS "/fp:strict")
    endif ()
endif ()

if (SAIGA_FAST_MATH)
    if (SAIGA_CXX_FLAGS_GNU)
        list(APPEND SAIGA_CXX_FLAGS "-ffast-math")
    endif ()
endif ()

######### debug and optimization #########

if (SAIGA_CUDA_DEBUG)
    add_definitions(-DCUDA_DEBUG)
else ()
    add_definitions(-DCUDA_NDEBUG)
endif ()

if (SAIGA_FULL_OPTIMIZE OR SAIGA_ARCHNATIVE)
    if (SAIGA_CXX_FLAGS_GNU)
        list(APPEND SAIGA_CXX_FLAGS "-march=native")
    endif ()
endif ()

if (SAIGA_FULL_OPTIMIZE)
    if (SAIGA_CXX_FLAGS_MSVC)
        set(SAIGA_CXX_FLAGS "${SAIGA_CXX_FLAGS} /Oi /Ot /Oy /GL /fp:fast /Gy")
        set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} /LTCG")
        add_definitions(-D__FMA__)
    endif ()
endif ()

# Sanitizers
# https://github.com/google/sanitizers
if (SAIGA_DEBUG_ASAN)
    if (SAIGA_CXX_FLAGS_GNU)
        SET(CMAKE_BUILD_TYPE Debug)
        list(APPEND SAIGA_CXX_FLAGS "-fsanitize=address")
        list(APPEND SAIGA_LD_FLAGS "-fsanitize=address")
        list(APPEND SAIGA_CXX_FLAGS "-fno-omit-frame-pointer")
    else ()
        message(FATAL_ERROR "ASAN not supported for your compiler.")
    endif ()
endif ()

if (SAIGA_DEBUG_MSAN)
    if (SAIGA_CXX_FLAGS_GNU)
        SET(CMAKE_BUILD_TYPE Debug)
        list(APPEND SAIGA_CXX_FLAGS "-fsanitize=memory")
        list(APPEND SAIGA_LD_FLAGS "-fsanitize=memory")
        list(APPEND SAIGA_CXX_FLAGS "-fno-omit-frame-pointer")
    else ()
        message(FATAL_ERROR "MSAN not supported for your compiler.")
    endif ()
endif ()

if (SAIGA_DEBUG_TSAN)
    if (SAIGA_CXX_FLAGS_GNU)
        list(APPEND SAIGA_CXX_FLAGS "-fsanitize=thread -static-libasan -fno-omit-frame-pointer -g")
    else ()
        message(FATAL_ERROR "TSAN not supported for your compiler.")
    endif ()
endif ()


