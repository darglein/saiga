if(CUDA_FOUND)
	add_subdirectory(runTests)
endif ()

if (SDL2_FOUND)
	add_subdirectory(simpleSDLWindow)
	add_subdirectory(advancedSDLWindow)
	add_subdirectory(cascadedShadowMaps)
	add_subdirectory(videoRecording)
endif ()

if (EGL_FOUND)
	add_subdirectory(offscreenRendering)
endif ()
