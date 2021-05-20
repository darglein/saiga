#set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(MODULE_NAME "saiga_vulkan")


FILE(GLOB_RECURSE VULKAN_SRC  "*.cpp" "*.cu" "*.cc")
FILE(GLOB_RECURSE VULKAN_HEADER  "*.h" "*.hpp" "*.inl" "Vulkan")


# the internal files are added to core
SET(SAIGA_ALL_FILES
    ${VULKAN_SRC}
    ${VULKAN_HEADER}
    )


if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
endif()

target_link_libraries(${MODULE_NAME} PUBLIC ${VULKAN_TARGETS}  ${VULKAN_LIBS})
target_compile_options(${MODULE_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_PRIVATE_CXX_FLAGS}>")

target_include_directories(
    ${MODULE_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:share/saiga/shader/include>
    #"$<INSTALL_INTERFACE:${VULKAN_INCLUDES}>"
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
    "$<BUILD_INTERFACE:${VULKAN_INCLUDES}>"
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/shader/include>
    #PRIVATE
    # for the internal files
    #${PROJECT_SOURCE_DIR}/src/
    )

install(
    TARGETS ${MODULE_NAME}
    EXPORT ${LIB_EXPORT}
    RUNTIME DESTINATION ${INSTALL_BIN}
    LIBRARY DESTINATION ${INSTALL_SHARED}
    ARCHIVE DESTINATION ${INSTALL_LIB}
    )

if(SAIGA_PCH)
    target_precompile_headers(
        ${MODULE_NAME}
      PRIVATE
        "${PROJECT_SOURCE_DIR}/src/internal/pch_vulkan.h"
    )
endif()
