# SAIGA

SAIGA is a lightweight OpenGL utitliy and rendering framework. It was successfully used as a game engine for [Redie](http://store.steampowered.com/app/536990/) and in many privat and university projects.

## History

 * 2014 January: Development start as a private OpenGL framework for university projects
 * 2015 September: The name SAIGA was chosen as a reference to the [saiga antelope](https://en.wikipedia.org/wiki/Saiga_antelope).
 * 2014 April - 2016 December: Development of the game [Redie](http://store.steampowered.com/app/536990/). In this time most of SAIGA's functionality was added so it could be used as a lightweight game engine.
 * 2017 January - 2017 August: Increased usability and documentation in preparation of the open source release
 * 2017 9. August: SAIGA goes Open-Source

## Supported Compilers

 * gcc 5.4 and newer
 * Visual Studio 2015 and newer
 * TODO: clang

## Supported Operating Systems

 * Ubuntu 16 64 bit and similiar Linux distros. (Older Linux systems should work by installing one of the supported compilers and building the dependencies from source.)
 * Windows Vista/7/8/10 32 and 64 bit

## Supported Graphics Hardware

 * All NVIDIA and AMD cards supporting atleast OpenGL 3.3.
 * Intel on chip graphic processor with linux mesa drivers. TODO: for windows
 * For the CUDA samples a NVIDIA GPU with compute capability 3 or newer is required.

## Required Dependencies

 * OpenGL 3.3 or newer
 * [GLEW](https://github.com/nigels-com/glew) or [glbinding](https://github.com/cginternals/glbinding)
 * [GLM](https://github.com/g-truc/glm)
 * [freetype](https://www.freetype.org/)

## Optional Dependencies

Window creation and GL-Context managment
 * [SDL](https://www.libsdl.org/)
 * [GLFW](http://www.glfw.org/)
 * [Mesa EGL](https://www.mesa3d.org/egl.html)
 
Sound loading and playback
 * [OpenAL](https://openal.org/)
 * [ALUT](http://distro.ibiblio.org/rootlinux/rootlinux-ports/more/freealut/freealut-1.1.0/doc/alut.html)
 * [Opus](http://opus-codec.org/)
 
Video Recording
 * [FFMPEG](https://ffmpeg.org/)
 
Model Loading
 * [ASSIMP](https://github.com/assimp/assimp)
 
Image loading
 * [PNG](http://www.libpng.org/pub/png/libpng.html)
 * [FreeImage + FreeImagePlus](http://freeimage.sourceforge.net/)
 
Utility
 * [Eigen](http://eigen.tuxfamily.org)
 * [CUDA](https://developer.nvidia.com/cuda-downloads)

## License

SAIGA is licensed under the MIT License. See LICENSE file for more information.

