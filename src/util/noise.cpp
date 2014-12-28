

#include "util/noise.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

using namespace noise;

Noise::Noise()
{
    module::Perlin myModule;
    double value = myModule.GetValue (1.25, 0.75, 0.50);

    std::cout << value << std::endl;
}
