set(MODULE_NAME "saiga_engine")

FILE(GLOB_RECURSE ENGINE_SRC "*.cpp")
FILE(GLOB_RECURSE ENGINE_HEADER "*.h")

SET(SAIGA_ALL_FILES
    ${ENGINE_SRC}
    ${ENGINE_HEADER}
    )

if(SAIGA_BUILD_SHARED)
    add_definitions(-DSAIGA_DLL_EXPORTS)
    add_library(${MODULE_NAME} SHARED ${SAIGA_ALL_FILES})
else ()
    add_library(${MODULE_NAME} STATIC ${SAIGA_ALL_FILES})
endif()
saiga_add_version_resource(${MODULE_NAME} "Saiga Engine")

# The content generation code with trapping UBSan (SAIGA_CONTENT_UBSAN, S23a; the Clang CI job)
if (SAIGA_CONTENT_UBSAN)
    FILE(GLOB_RECURSE ENGINE_CONTENT_SRC "content/*.cpp")
    set_source_files_properties(${ENGINE_CONTENT_SRC} PROPERTIES COMPILE_OPTIONS "${SAIGA_CONTENT_UBSAN_FLAGS}")
endif ()

target_link_libraries(${MODULE_NAME} PUBLIC ${ENGINE_TARGETS} ${ENGINE_LIBS})
target_compile_options(${MODULE_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SAIGA_PRIVATE_CXX_FLAGS}>")

target_include_directories(
    ${MODULE_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
    "$<BUILD_INTERFACE:${ENGINE_INCLUDES}>"
    )

install(
    TARGETS ${MODULE_NAME}
    EXPORT ${LIB_EXPORT}
    RUNTIME DESTINATION ${INSTALL_BIN}
    LIBRARY DESTINATION ${INSTALL_SHARED}
    ARCHIVE DESTINATION ${INSTALL_LIB}
    )
