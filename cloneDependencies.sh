mkdir dependencies
cd dependencies

git clone https://github.com/g-truc/glm.git
git clone https://github.com/aseprite/freetype2.git

git clone https://github.com/nigels-com/glew.git
cd glew
git checkout tags/glew-2.1.0
cd ..

git clone https://github.com/spurious/SDL-mirror.git sdl2
git clone https://github.com/glfw/glfw.git
git clone https://github.com/KhronosGroup/glslang.git
git clone https://github.com/glennrp/libpng.git png
git clone https://github.com/eigenteam/eigen-git-mirror.git eigen
git clone https://github.com/assimp/assimp.git
git clone https://github.com/madler/zlib.git
#freeimage: http://freeimage.sourceforge.net/download.html