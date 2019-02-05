#set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(MODULE_NAME "saiga_vision")


add_subdirectory(ba)
add_subdirectory(icp)
add_subdirectory(kernels)
add_subdirectory(pgo)
add_subdirectory(recursiveMatrices)
add_subdirectory(scene)



if(OPENNI2_FOUND)
    add_subdirectory(openni2)
endif()

if(G2O_FOUND)
    add_subdirectory(g2o)
endif()

if(CERES_FOUND)
    add_subdirectory(ceres)
endif()

FILE(GLOB VULKAN_SRC  "*.cpp" "*.cu" "*.cc")
FILE(GLOB VULKAN_HEADER  "*.h" "*.hpp" "*.inl")

SET(SAIGA_ALL_FILES
    ${SAIGA_ALL_FILES}
    ${VULKAN_SRC}
    ${VULKAN_HEADER}
    )



if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
endif()

target_link_libraries(${MODULE_NAME}  saiga_core )
target_link_libraries(${MODULE_NAME} ${ALL_CUDA_LIBS}  )
target_link_libraries(${MODULE_NAME}  ${LIBS} )
target_link_libraries(${MODULE_NAME} ${LIB_TARGETS} )

target_include_directories(
    ${MODULE_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/saiga/opengl/glbinding/include>
    $<INSTALL_INTERFACE:include/saiga/opengl/glbinding-aux/include>
    $<INSTALL_INTERFACE:share/saiga/shader/include>
    #"$<INSTALL_INTERFACE:${PACKAGE_INCLUDES}>"
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/saiga/opengl/glbinding/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/saiga/opengl/glbinding-aux/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/shader/include>
    "$<BUILD_INTERFACE:${PACKAGE_INCLUDES}>"
    #PRIVATE
    # for the internal files
    #${PROJECT_SOURCE_DIR}/src/
    )

##### Precompiled Headers #####
if(SAIGA_PCH)
    set_target_properties(${MODULE_NAME} PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "${PROJECT_SOURCE_DIR}/src/internal/pch_vision.h")
    set_target_properties(${MODULE_NAME} PROPERTIES COTIRE_ADD_UNITY_BUILD false)
    cotire(${MODULE_NAME})
endif()
