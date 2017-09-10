# SAIGA

SAIGA is a lightweight OpenGL utility and rendering framework. It was successfully used as a game engine for [Redie](http://store.steampowered.com/app/536990/) and in many privat and university projects.

<img src="bin/textures/sample.png" width="425"/> <img src="bin/textures/redie.jpg" width="425"/> 

## History

 * January 2014: Development start as a private OpenGL framework for university projects.
 * September 2015: The name SAIGA was chosen as a reference to the [saiga antelope](https://en.wikipedia.org/wiki/Saiga_antelope).
 * April 2014 - December 2016: Development of the game [Redie](http://store.steampowered.com/app/536990/). In this time most of SAIGA's functionality was added so it could be used as a lightweight game engine.
 * January 2017 - August 2017: Increased usability and documentation in preparation of the open source release.
 * August 2017: Open-Source Release.

## Supported Compilers

 * g++ 5.4 or newer
 * Visual Studio 2013 or newer
 * clang++ 3.8 or newer

## Supported Operating Systems

 * Ubuntu 16 64 bit and similiar Linux distros
 * Fedora 25 64 bit or newer

   Other Linux systems should work by installing one of the supported compilers and building the dependencies from source.
 * Windows Vista/7/8/10 32 and 64 bit

## Supported Graphics Hardware

 * All NVIDIA and AMD cards supporting atleast OpenGL 3.3.
 * Intel on chip graphic processor with linux mesa drivers or similiar windows driver.
 * For the CUDA samples a NVIDIA GPU with compute capability 3 or newer is required.

## Required Dependencies

 * OpenGL 3.3
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

## Building + Running the samples

#### Linux
 - Install dependencies with the package manager (in older Linux systems you might have to compile the latest library versions by yourself)

   For Ubuntu and other Debian-based distributions:
   ```
   sudo apt-get install libglew-dev libglm-dev libfreetype6-dev libsdl2-dev libpng-dev
   ```
   For Fedora:
   ```
   sudo dnf install glew-devel glm-devel freetype-devel SDL2-devel libXrandr-devel libXcursor-devel libXinerama-devel
   ```
 - Build Saiga
```
cd saiga
mkdir build
cd build
cmake ..
make
```
 - Running the samples
```
cd saiga/bin
./simpleSDLWindow
```

#### Windows
 - Download and compile the dependencies from source. (For a quick start only get glew,glm,freetype,sdl2 and png).
 - Install the dependencies or copy them to a common location with the following structure:
```
<your_dependencies_dir>
<your_dependencies_dir>/include      <- Put all header files here
<your_dependencies_dir>/lib          <- Put all .lib files here
<your_dependencies_dir>/bin          <- Put all .dll files here
```
Use cmake to create the Visual Studio solution with the following cmake variable set:
```
DEPENDENCIES_DIR=<your_dependencies_dir>
```
 - Compile the solution with Visual Studio. 
 - saiga.dll and all executables will be placed for example in saiga/bin/RelWithDebugInfo. 
 - When running the samples make sure the working directory is saiga/bin instead of saiga/bin/RelWithDebugInfo.

## License

SAIGA is licensed under the MIT License. See LICENSE file for more information.


