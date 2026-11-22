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

# SAIGA_WARNINGS_AS_ERRORS: added by saiga_core to its build interface (the libraries and everything in this build that
# links them: tests, samples); the submodules are not affected.
set(SAIGA_WERROR_CXX_FLAGS)
set(SAIGA_WERROR_LD_FLAGS)
if (SAIGA_WARNINGS_AS_ERRORS)
    if (SAIGA_CXX_FLAGS_GNU)
        list(APPEND SAIGA_WERROR_CXX_FLAGS "-Werror")
    endif ()
    if (SAIGA_CXX_FLAGS_MSVC)
        list(APPEND SAIGA_WERROR_CXX_FLAGS "/WX")
    endif ()
    # The linker flag follows the linker, not the compiler frontend: Clang with the GNU frontend targeting the MSVC ABI
    # links with lld-link (MSVC options)
    if (MSVC OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
        list(APPEND SAIGA_WERROR_LD_FLAGS "LINKER:/WX")
    elseif (NOT APPLE)
        list(APPEND SAIGA_WERROR_LD_FLAGS "LINKER:--fatal-warnings")
    endif ()
endif ()

# SAIGA_CONTENT_UBSAN (S23a, docs/engine_design.md §12.1): trapping UBSan for the content generation code only (the
# sources under src/saiga/engine/content/ and test_engine_content). Trapping needs no sanitizer runtime, so the shared
# library and every other target link as before; undefined behaviour aborts the test (SIGILL).
set(SAIGA_CONTENT_UBSAN_FLAGS)
if (SAIGA_CONTENT_UBSAN)
    if (MSVC)
        message(FATAL_ERROR "SAIGA_CONTENT_UBSAN needs Clang or GCC")
    endif ()
    set(SAIGA_CONTENT_UBSAN_FLAGS
        "-fsanitize=signed-integer-overflow,shift,integer-divide-by-zero,bounds,null,alignment,bool,enum,return,unreachable"
        "-fsanitize-trap=all")
    include(CheckCXXCompilerFlag)
    string(REPLACE ";" " " SAIGA_CONTENT_UBSAN_FLAGS_STRING "${SAIGA_CONTENT_UBSAN_FLAGS}")
    check_cxx_compiler_flag("${SAIGA_CONTENT_UBSAN_FLAGS_STRING}" SAIGA_CONTENT_UBSAN_SUPPORTED)
    if (NOT SAIGA_CONTENT_UBSAN_SUPPORTED)
        message(FATAL_ERROR "SAIGA_CONTENT_UBSAN: the compiler does not accept ${SAIGA_CONTENT_UBSAN_FLAGS} (Clang, or GCC 13 or newer)")
    endif ()
    message(STATUS "SAIGA_CONTENT_UBSAN: ${SAIGA_CONTENT_UBSAN_FLAGS}")
endif ()

#if (SAIGA_CXX_WCLANG)
# Fixes: cannot use 'throw' with exceptions disabled
#list(APPEND SAIGA_CXX_FLAGS "-Xclang -fcxx-exceptions")
# some eigen header generates this warning
#    add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
#endif ()


if (SAIGA_CXX_FLAGS_MSVC)

    # Multiprocessor compilation for the Visual Studio generators (MSBuild hands cl several files at once). Ninja
    # compiles one file per cl call, where /MP does nothing and keeps compiler caches (ccache in CI) from caching.
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" AND CMAKE_GENERATOR MATCHES "Visual Studio")
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

# SAIGA_ARCHNATIVE_ACTIVE: -march=native is really used (only GCC-style compilers; saiga_buildconfig.h)
set(SAIGA_ARCHNATIVE_ACTIVE OFF)
if (SAIGA_FULL_OPTIMIZE OR SAIGA_ARCHNATIVE)
    if (SAIGA_CXX_FLAGS_GNU)
        list(APPEND SAIGA_CXX_FLAGS "-march=native")
        set(SAIGA_ARCHNATIVE_ACTIVE ON)
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
        list(APPEND SAIGA_CXX_FLAGS "-fsanitize=address,undefined")
        list(APPEND SAIGA_LD_FLAGS "-fsanitize=address,undefined")
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


