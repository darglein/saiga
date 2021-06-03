set(MODULE_NAME "saiga_cuda")




# ==== Load All Files ====

FILE(GLOB_RECURSE CUDA_SRC  "*.cpp" "*.cu" "*.cc")
FILE(GLOB_RECURSE CUDA_HEADER  "*.h" "*.hpp" "*.inl")

# the internal files are added to core
SET(SAIGA_ALL_FILES
    ${CUDA_SRC}
    ${CUDA_HEADER}
    )


if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
endif()
target_compile_options(${MODULE_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_PRIVATE_CXX_FLAGS}>")
target_link_libraries(${MODULE_NAME} ${ALL_CUDA_LIBS})
target_link_libraries(${MODULE_NAME} ${CUDA_LIBS})
target_link_libraries(${MODULE_NAME} ${CUDA_TARGETS})
#set_target_properties(${MODULE_NAME} PROPERTIES CXX_STANDARD 17)

#list(APPEND SAIGA_CUDA_FLAGS "-std=c++14")



set_property(TARGET ${MODULE_NAME} PROPERTY CUDA_ARCHITECTURES ${SAIGA_CUDA_ARCH})
target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${SAIGA_CUDA_FLAGS}>)

target_include_directories(
    ${MODULE_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
    "$<BUILD_INTERFACE:${CUDA_INCLUDES}>"
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

