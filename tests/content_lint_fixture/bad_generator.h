/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The ContentLint self-test (S23a): every refused token once in code, the same tokens in comments and literals, and
// allowed look-alikes. Not compiled. Lines 41 and 42 end with CRLF; line 43 continues on line 44.
#include <cmath>
#include "saiga/engine/world/SpaceTemplate.h"
#include <vector>
float a;
double b;
long c;
size_t d;
__int128 e;
std::unordered_map<int, int> f;
std::uniform_int_distribution<int> g;
int h = rand();
Rng i;
int j = sin(1);
int k = std::floor(2);
std::hash<int> l;
int m = kHeightUnit;
int n = terrainSurfaceDouble(0, 0, 0, 0, 0, 0);
int o = chunkSize + chunkOrigin;
Vec3 p; quat q;
SpaceTemplate r;
int s = 0.7;
int t = 1e3;
int u = .5f;
int v = 2.;
int w = 0x1p3;
/* float double long size_t sin( kHeightUnit 0.5 #include <cmath> */
// SpaceTemplate Rng std::hash 1e3
const char* x = "float double sqrt( 0.5 SpaceTemplate";
const char* y = R"d(float " double )" 1e3 sin()d";
char z = 'f';
int aa = 0x1e3 + 10'000 + floorDiv(1, 2) + 65536u + isqrt64(4);
int ab = pow(2, 3);
double ac = 1;
int ad = 3; // fine	
int ae = std::cbr\
t(8);
int af = 0;
