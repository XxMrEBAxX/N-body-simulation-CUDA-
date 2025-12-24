#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <vector>
#include "Vector.h"

const double TAU = 2.0 * M_PI;

struct Body {
    NBody::Vector2 position;
    NBody::Vector2 velocity;
    NBody::Vector2 acceleration;
    double mass;
};

class BodyGenerator
{
public:
    static std::vector<Body> generateRandomBodies(int count, int width, int height);
	static std::vector<Body> generateSpiralBodies(int count, int width, int height);
	static std::vector<Body> generateSolarSystemBodies();
};

