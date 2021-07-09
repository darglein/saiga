macro(saiga_opengl_sample FILE_NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

  set(TARGET_NAME ${NAME})

  add_executable(${TARGET_NAME} ${FILE_NAME})
  message(STATUS "Sample enabled:      ${NAME}")


  target_link_libraries(${TARGET_NAME} PUBLIC saiga_core saiga_opengl)
  target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})

  set_target_properties(${TARGET_NAME} PROPERTIES FOLDER samples/${PREFIX})
endmacro()




if (GLFW_FOUND)
  saiga_opengl_sample(sample_opengl_cascaded_shadow_maps.cpp)
  saiga_opengl_sample(sample_opengl_cornell.cpp)
  saiga_opengl_sample(sample_opengl_forward_rendering.cpp)
  saiga_opengl_sample(sample_opengl_lighting.cpp)
  saiga_opengl_sample(sample_opengl_model_viewer.cpp)
  saiga_opengl_sample(sample_opengl_window.cpp)
  saiga_opengl_sample(sample_opengl_split_screen.cpp)
  saiga_opengl_sample(sample_opengl_volumetric_lights.cpp)
  saiga_opengl_sample(sample_opengl_shapes.cpp)
  saiga_opengl_sample(sample_opengl_particles.cpp)
  saiga_opengl_sample(sample_opengl_hdr.cpp)
  saiga_opengl_sample(sample_opengl_bloom.cpp)
  saiga_opengl_sample(sample_opengl_enviroment_mapping.cpp)

  if (ASSIMP_FOUND)
    saiga_opengl_sample(sample_opengl_bone_animation.cpp)
  endif ()

  if (BULLET_FOUND)
    add_subdirectory(simpleBullet)
  endif ()
  if (OPENMESH_FOUND)
    add_subdirectory(depthMapTriangulation)
  endif ()

  if(FREETYPE_FOUND)
    saiga_opengl_sample(sample_opengl_text.cpp)
  endif()

endif ()

if (EGL_FOUND)
  saiga_opengl_sample(sample_opengl_egl.cpp)
endif ()

if(OPENVR_FOUND)
  saiga_opengl_sample(sample_opengl_vr.cpp)
endif()
