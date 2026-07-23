

message("=================================")
if (SAIGA_WITH_TINY_EIGEN)
    message("Adding Submodule tiny-eigen")
    add_subdirectory(tiny-eigen)
else ()
    message("Adding Submodule eigen")
    set(BUILD_TESTING OFF CACHE INTERNAL "")
    add_subdirectory(eigen)
    set(SAIGA_USE_EIGEN 1 PARENT_SCOPE)
endif ()
message("=================================")


if (SAIGA_WITH_TINY_GLOG)
    message("Adding Submodule tiny-glog")
    add_subdirectory(tiny-glog)
else ()
    message("Adding Submodule glog")
    set(WITH_GFLAGS OFF CACHE INTERNAL "")
    set(WITH_GTEST OFF CACHE INTERNAL "")
    set(WITH_UNWIND OFF CACHE INTERNAL "")
    set(BUILD_TESTING OFF CACHE INTERNAL "")
    add_subdirectory(glog)
    SET(SAIGA_USE_GLOG 1 PARENT_SCOPE)
endif ()

message("=================================")


message("=================================")
message("Adding Submodule ZLIB")


set(ZLIB_BUILD_SHARED OFF CACHE INTERNAL "")
set(ZLIB_BUILD_STATIC ON CACHE INTERNAL "")
set(ZLIB_BUILD_EXAMPLES OFF CACHE INTERNAL "")

# Need this because for some reason assimps exports zlib as well
if (SAIGA_WITH_ASSIMP)
    set(SKIP_INSTALL_ALL ON CACHE INTERNAL "")
endif ()
set(ZLIB_TARGET zlibstatic)
add_subdirectory(zlib)
set(SKIP_INSTALL_ALL OFF CACHE INTERNAL "")

# libPNG uses include_directory(ZLIB_INCLUDE_DIR) therefore we have to set these variables
set(ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib CACHE PATH "zlib dir" FORCE)
set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR} CACHE PATH "zlib dir" FORCE)

# include dir that use the zlib target
target_include_directories(${ZLIB_TARGET} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/zlib>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/zlib>)

#set_target_properties(${ZLIB_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
SET(SAIGA_USE_ZLIB 1 PARENT_SCOPE)
SET(ZLIB_FOUND 1)
set(ZLIB_LIBRARIES ${ZLIB_TARGET} CACHE INTERNAL "")
set(ZLIB_LIBRARY zlibstatic CACHE INTERNAL "")
add_library(ZLIB::ZLIB ALIAS ${ZLIB_TARGET})


# Create a fake zlib target which also points to the zlib static library
add_library(zlib INTERFACE)
target_link_libraries(zlib INTERFACE zlibstatic)

message("=================================")


message("=================================")
message("Adding Submodule zstd")
set(ZSTD_BUILD_SHARED OFF CACHE INTERNAL "")
set(ZSTD_BUILD_STATIC ON CACHE INTERNAL "")
set(ZSTD_BUILD_PROGRAMS OFF CACHE INTERNAL "")
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(zstd/build/cmake/)

if (SAIGA_BUILD_GLFW)
    message("=================================")
    message("Adding Submodule glfw")

    if (SAIGA_GLFW_STATIC)
        set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    else ()
        set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
    endif ()

    set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
    set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")

    add_subdirectory(${PROJECT_SOURCE_DIR}/submodules/glfw)
    #    set_target_properties(glfw PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
    SET(SAIGA_USE_GLFW 1 PARENT_SCOPE)

    #find_package(glfw3 CONFIG QUIET REQUIRED)
    SET(SAIGA_USE_GLFW 1)
    #    PackageHelperTarget(glfw GLFW_FOUND)
    message("=================================")
endif ()


message("=================================")
message("Adding Submodule libPNG")

#set(PNG_BUILD_ZLIB ON CACHE INTERNAL "")
set(PNG_STATIC ON CACHE INTERNAL "")
set(PNG_SHARED OFF CACHE INTERNAL "")
#set(PNG_EXECUTABLES OFF CACHE INTERNAL "")
set(PNG_TOOLS OFF CACHE INTERNAL "")
set(PNG_TESTS OFF CACHE INTERNAL "")

include_directories(${ZLIB_INCLUDE_DIRS})

add_subdirectory(libpng)

set_property(TARGET png_static PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(png_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/libpng>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/libpng>)

#set_target_properties(png_static PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")
SET(SAIGA_USE_PNG 1 PARENT_SCOPE)

#  set(CMAKE_INSTALL_LIBDIR lib)
#    install(TARGETS png zlib
#          EXPORT libpng
#          RUNTIME DESTINATION bin
#          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#          FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR})
#
#  install(EXPORT libpng
#          DESTINATION lib/libpng)

message("=================================")


if (SAIGA_WITH_ASSIMP)
    message("=================================")
    message("Adding Submodule assimp")

    set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "")
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE INTERNAL "")

    add_subdirectory(assimp)
    #set_target_properties(assimp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SAIGA_RUNTIME_OUTPUT_DIRECTORY}")


    SET(SAIGA_USE_ASSIMP 1 PARENT_SCOPE)
    message("=================================")
endif ()


if (SAIGA_WITH_GTEST)
    message("=================================")
    message("Adding Submodule googletest")
    set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
    add_subdirectory(googletest)
    set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
    PackageHelperTarget(gtest GTEST_FOUND)
    set(GTEST_FOUND 1 PARENT_SCOPE)
    set(GTEST_TARGETS gtest gtest_main)
    message("=================================")
endif ()


if (SAIGA_WITH_LIBZIP)
    message("=================================")
    message("Adding Submodule libzip")

    set(BUILD_SHARED_LIBS OFF)

    set(ENABLE_COMMONCRYPTO OFF CACHE INTERNAL "")
    set(ENABLE_GNUTLS OFF CACHE INTERNAL "")
    set(ENABLE_MBEDTLS OFF CACHE INTERNAL "")
    set(ENABLE_OPENSSL OFF CACHE INTERNAL "")
    set(ENABLE_WINDOWS_CRYPTO OFF CACHE INTERNAL "")
    set(ENABLE_BZIP2 OFF CACHE INTERNAL "")
    set(ENABLE_LZMA OFF CACHE INTERNAL "")




    # Disable tools
    set(BUILD_TOOLS OFF CACHE INTERNAL "")
    set(BUILD_REGRESS OFF CACHE INTERNAL "")
    set(BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(BUILD_DOC OFF CACHE INTERNAL "")
    set(BUILD_OSSFUZZ OFF CACHE INTERNAL "")

    add_subdirectory(libzip)
    set_property(TARGET zip PROPERTY POSITION_INDEPENDENT_CODE ON)

    SET(SAIGA_USE_LIBZIP 1 PARENT_SCOPE)
    message("=================================")
endif ()


if (SAIGA_WITH_LIBTIFF)
    message("=================================")
    message("Adding Submodule libtiff")

    set(tiff-tools OFF CACHE INTERNAL "")
    set(tiff-tools-unsupported OFF CACHE INTERNAL "")
    set(tiff-tests OFF CACHE INTERNAL "")
    set(tiff-contrib OFF CACHE INTERNAL "")
    set(tiff-docs OFF CACHE INTERNAL "")
    set(tiff-deprecated OFF CACHE INTERNAL "")
    set(tiff-install OFF CACHE INTERNAL "")
    set(zlib ON CACHE INTERNAL "")

    macro(install)
    endmacro()

    add_subdirectory(libtiff)

    macro(install)
        _install(${ARGN})
    endmacro()

    set_property(TARGET tiff PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET tiffxx PROPERTY POSITION_INDEPENDENT_CODE ON)

    # this exe can't be disable -> write it to a different dir
    #set_target_properties(tiff_mkg3states PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin_tiff")

    SET(SAIGA_USE_LIBTIFF 1 PARENT_SCOPE)
    message("=================================")
endif ()

if (SAIGA_WITH_TINYTIFF)
    message("=================================")
    message("Adding Submodule TinyTIFF")

    set(TinyTIFF_BUILD_SHARED_LIBS OFF)
    set(TinyTIFF_BUILD_STATIC_LIBS ON)
    set(TinyTIFF_BUILD_TESTS OFF)

    add_subdirectory(TinyTIFF)
    set_property(TARGET TinyTIFF PROPERTY POSITION_INDEPENDENT_CODE ON)

    SET(SAIGA_USE_TINYTIFF 1 PARENT_SCOPE)
    message("=================================")
endif ()


if (SAIGA_WITH_LIBAVIF)
    message(STATUS "=================================")
    message(STATUS "Configuring libavif")

    # Global options for libavif
    set(AVIF_LIBYUV OFF CACHE STRING "" FORCE)
    set(AVIF_BUILD_APPS OFF CACHE BOOL "" FORCE)
    set(AVIF_BUILD_TESTS OFF CACHE BOOL "" FORCE)

    # -------------------------------------------------------------
    # 1. SVT-AV1 Codec Configuration
    # -------------------------------------------------------------
    if (SAIGA_WITH_SVTAV1)
        message(STATUS "Adding Submodule SVT-AV1")

        set(ENABLE_LTO OFF CACHE BOOL "Disable LTO for SVT" FORCE)
        set(BUILD_APPS OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(SVT_AV1_LTO OFF CACHE BOOL "Disable LTO for SVT-AV1" FORCE)
        set(ENABLE_AVX512 OFF CACHE BOOL "Disable LTO for SVT-AV1" FORCE)

        add_subdirectory(SVT-AV1)

        # Header wrapper hack to satisfy libavif's expected include layout
        set(SVT_WRAPPER_DIR "${CMAKE_CURRENT_BINARY_DIR}/svt_wrapper_includes")
        file(MAKE_DIRECTORY "${SVT_WRAPPER_DIR}/svt-av1")
        file(GLOB SVT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/SVT-AV1/Source/API/*.h")
        file(COPY ${SVT_HEADERS} DESTINATION "${SVT_WRAPPER_DIR}/svt-av1/")

        target_include_directories(SvtAv1Enc INTERFACE "${SVT_WRAPPER_DIR}")

        set(AVIF_CODEC_SVT SYSTEM CACHE STRING "" FORCE)
        set(AVIF_LOCAL_SVT OFF CACHE BOOL "" FORCE)
        set(SVT_AV1_FOUND ON CACHE BOOL "" FORCE)
        set(SVT_AV1_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SVT-AV1/Source/API" CACHE PATH "" FORCE)
        set(SVT_AV1_LIBRARY SvtAv1Enc CACHE STRING "" FORCE)
        
        set(SAIGA_USE_SVTAV1 1 PARENT_SCOPE)
    else ()
        set(AVIF_CODEC_SVT OFF CACHE STRING "" FORCE)
        set(AVIF_LOCAL_SVT OFF CACHE BOOL "" FORCE)
    endif ()

    # -------------------------------------------------------------
    # 2. libaom Codec Configuration
    # -------------------------------------------------------------
    if (SAIGA_WITH_LIBAOM)
        message(STATUS "Adding Submodule aom")

        # Locate Perl for libaom if not automatically found on Windows
        if (WIN32 AND NOT PERL_EXECUTABLE)
            if (EXISTS "C:/Program Files/Git/usr/bin/perl.exe")
                set(PERL_EXECUTABLE "C:/Program Files/Git/usr/bin/perl.exe" CACHE FILEPATH "" FORCE)
            endif ()
        endif ()

        # Locate NASM & Inject into runtime PATH
        if (NOT AS_EXECUTABLE OR NOT EXISTS "${AS_EXECUTABLE}")
            find_program(AS_EXECUTABLE NAMES nasm nasm.exe yasm yasm.exe
                PATHS
                "C:/Program Files/NASM"
                "C:/Program Files (x86)/NASM"
                "$ENV{LOCALAPPDATA}/bin/NASM"
                "$ENV{LOCALAPPDATA}/Programs/NASM"
                "$ENV{ProgramData}/chocolatey/bin"
            )
        endif ()

        if (AS_EXECUTABLE)
            message(STATUS "Found NASM assembler for libaom: ${AS_EXECUTABLE}")
            get_filename_component(NASM_DIR "${AS_EXECUTABLE}" DIRECTORY)
            file(TO_CMAKE_PATH "${NASM_DIR}" NASM_DIR)
            set(ENV{PATH} "${NASM_DIR};$ENV{PATH}")
        else ()
            message(FATAL_ERROR "NASM executable (nasm.exe) not found! Install NASM or set -DAS_EXECUTABLE='path/to/nasm.exe'")
        endif ()

        # --- SANDBOX FUNCTION TO PREVENT LIBAOM FLAG/CACHE POLLUTION ---
        function(build_aom_isolated)
            # 1. Save ALL C and CXX flags (base + configuration variants)
            foreach(lang C CXX)
                foreach(config "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
                    set(SAVED_${lang}_FLAGS${config} "${CMAKE_${lang}_FLAGS${config}}")
                endforeach()
            endforeach()
            set(SAVED_BUILD_SHARED "${BUILD_SHARED_LIBS}")

            # Set libaom cache variables
            set(ENABLE_NASM ON CACHE BOOL "Enable NASM assembly optimizations in libaom" FORCE)
            set(AS_EXECUTABLE "${AS_EXECUTABLE}" CACHE FILEPATH "Path to NASM/YASM assembler for libaom" FORCE)
            set(CMAKE_ASM_NASM_COMPILER "${AS_EXECUTABLE}" CACHE FILEPATH "NASM compiler" FORCE)

            set(ENABLE_DOCS OFF CACHE BOOL "" FORCE)
            set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
            set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
            set(ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
            set(ENABLE_VS_IDE_PROJECTS OFF CACHE BOOL "" FORCE)
            set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

            set(CONFIG_AV1_DECODER 0 CACHE STRING "" FORCE)

            # Fix target name collision ("dist") between libzip & libaom
            function(add_custom_target target_name)
                if("${target_name}" STREQUAL "dist")
                    _add_custom_target(aom_dist ${ARGN})
                else()
                    _add_custom_target(${target_name} ${ARGN})
                endif()
            endfunction()

            add_subdirectory(aom)

            # 2. Restore ALL flags in both CACHE and local scope
            foreach(lang C CXX)
                foreach(config "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
                    set(CMAKE_${lang}_FLAGS${config} "${SAVED_${lang}_FLAGS${config}}" CACHE STRING "" FORCE)
                    set(CMAKE_${lang}_FLAGS${config} "${SAVED_${lang}_FLAGS${config}}")
                endforeach()
            endforeach()
    
            set(BUILD_SHARED_LIBS "${SAVED_BUILD_SHARED}" CACHE BOOL "" FORCE)
            set(BUILD_SHARED_LIBS "${SAVED_BUILD_SHARED}")
        endfunction()

        build_aom_isolated()

        # Expose libaom include paths specifically and ONLY to libavif
        set(AOM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aom")
        set(AOM_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/aom")

        if (TARGET aom)
            target_include_directories(aom INTERFACE 
                "${AOM_SRC_DIR}"
                "${AOM_BIN_DIR}"
            )
        endif ()

        set(AVIF_CODEC_AOM SYSTEM CACHE STRING "" FORCE)
        set(AVIF_CODEC_AOM_ENCODE ON CACHE BOOL "" FORCE)
        set(AVIF_CODEC_AOM_DECODE OFF CACHE BOOL "" FORCE)
        set(AVIF_LOCAL_AOM OFF CACHE BOOL "" FORCE)

        set(AOM_FOUND ON CACHE BOOL "" FORCE)
        set(AOM_INCLUDE_DIR "${AOM_SRC_DIR};${AOM_BIN_DIR}" CACHE PATH "" FORCE)
        set(AOM_LIBRARIES aom CACHE STRING "" FORCE)
        
        set(SAIGA_USE_LIBAOM 1 PARENT_SCOPE)
    else ()
        set(AVIF_CODEC_AOM OFF CACHE STRING "" FORCE)
        set(AVIF_LOCAL_AOM OFF CACHE BOOL "" FORCE)
    endif ()

    # -------------------------------------------------------------
    # 3. Add libavif Submodule
    # -------------------------------------------------------------
    message(STATUS "Adding Submodule libavif")
    add_subdirectory(libavif)

    set(SAIGA_USE_LIBAVIF 1 PARENT_SCOPE)
    message(STATUS "=================================")
endif ()

