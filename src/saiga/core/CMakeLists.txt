set(MODULE_NAME "saiga_core")

add_subdirectory(camera)
add_subdirectory(framework)
add_subdirectory(math)
add_subdirectory(model)
add_subdirectory(window)
add_subdirectory(image)
add_subdirectory(time)
add_subdirectory(geometry)
add_subdirectory(util)
add_subdirectory(tests)
add_subdirectory(imgui)
add_subdirectory(rendering)
add_subdirectory(sophus)


if (GLFW_FOUND)
    add_subdirectory(glfw)
endif ()

if (OPENAL_FOUND)
    add_subdirectory(sound)
endif ()

# the internal files are added to core
SET(SAIGA_ALL_FILES
        ${SAIGA_ALL_FILES}
        ${SAIGA_INTERNAL_SOURCE}
        ${SAIGA_INTERNAL_HEADERS}
        ${SAIGA_CONFIGURED_FILES}
        )

include_directories(${PROJECT_SOURCE_DIR}/src/)

if (SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
    set_property(TARGET ${MODULE_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
endif ()

target_compile_options(${MODULE_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_CXX_FLAGS}>")
target_compile_options(${MODULE_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_PRIVATE_CXX_FLAGS}>")
target_link_options(${MODULE_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_LD_FLAGS}>")

# as a compile feature this flag propagates to all libraries that use saiga
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

target_link_libraries(${MODULE_NAME} PUBLIC ${CORE_LIBS} ${CORE_TARGETS})
#target_link_options(${MODULE_NAME} PUBLIC ${SAIGA_LD_FLAGS})
target_link_libraries(${MODULE_NAME} PRIVATE $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,8.0>>:stdc++fs>)

#message(FATAL_ERROR "saiga core libs: ${CORE_LIBS} | ${CORE_TARGETS}")

target_include_directories(
        ${MODULE_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/shader/include/>
        "$<BUILD_INTERFACE:${CORE_INCLUDES}>"
        "$<BUILD_INTERFACE:${SAIGA_BUILDCONFIG_DIR}>"

        #PRIVATE
        # for the internal files
        #${PROJECT_SOURCE_DIR}/src/
)

set(OTHER_EXPORT_TARGETS "")

install(
        TARGETS ${MODULE_NAME} ${OTHER_EXPORT_TARGETS}
        EXPORT ${LIB_EXPORT}
        RUNTIME DESTINATION ${INSTALL_BIN}
        LIBRARY DESTINATION ${INSTALL_SHARED}
        ARCHIVE DESTINATION ${INSTALL_LIB}
)


# https://www.qt.io/blog/2019/08/01/precompiled-headers-and-unity-jumbo-builds-in-upcoming-cmake
if (SAIGA_PCH)
    target_precompile_headers(
            ${MODULE_NAME}
            PRIVATE
            "${PROJECT_SOURCE_DIR}/src/internal/pch_core.h"
    )
endif ()
