set(MODULE_NAME "saiga_vision")


add_subdirectory(arap)
add_subdirectory(ba)
add_subdirectory(camera)
add_subdirectory(cameraModel)
add_subdirectory(icp)
add_subdirectory(imu)
add_subdirectory(kernels)
add_subdirectory(optimizationProblems)
add_subdirectory(pgo)
add_subdirectory(reconstruction)
add_subdirectory(scene)
add_subdirectory(slam)
add_subdirectory(util)
add_subdirectory(torch)
add_subdirectory(features)

if (OPENCV_FOUND)
    add_subdirectory(orb)
    add_subdirectory(opencv)
endif()

if(MKL_FOUND)
    add_subdirectory(mkl)
endif()

if(G2O_FOUND)
    add_subdirectory(g2o)
endif()

if(CERES_FOUND)
    add_subdirectory(ceres)
endif()

if(SAIGA_USE_EIGENRECURSIVE)
    add_subdirectory(recursive)
endif()

FILE(GLOB VISION_SRC  "*.cpp" "*.cu" "*.cc")
FILE(GLOB VISION_HEADER  "*.h" "*.hpp" "*.inl")

SET(SAIGA_ALL_FILES
    ${SAIGA_ALL_FILES}
    ${VISION_SRC}
    ${VISION_HEADER}
    )



if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
endif()

target_link_libraries(${MODULE_NAME} ${VISION_TARGETS}  ${VISION_LIBS})
target_compile_options(${MODULE_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_PRIVATE_CXX_FLAGS}>")

target_include_directories(
    ${MODULE_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>
    #"$<INSTALL_INTERFACE:${VISION_INCLUDES}>"
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
    "$<BUILD_INTERFACE:${VISION_INCLUDES}>"
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
        "${PROJECT_SOURCE_DIR}/src/internal/pch_vision.h"
    )
endif()
