#include "opengl/texture/freeimage.h"




Freeimage::Freeimage()
{
    FreeImage_Initialise(true);




}

void Freeimage::load(std::string filename)
{


    fipImage fipimg;
    fipimg.load(filename.c_str());

    cout << "Image: " << filename << " is size: " << fipimg.getWidth() << "x" << fipimg.getHeight() << " colors " << fipimg.getColorsUsed() << " bits per pixel "<<fipimg.getBitsPerPixel()<< " type"<<fipimg.getColorType()<< endl;




}
