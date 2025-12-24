#include "Body.h"

#include <random>

std::vector<Body> BodyGenerator::generateRandomBodies(int count, int width, int height)
{
	std::vector<Body> bodies(count);
	for (int i = 0; i < count; i++) {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_int_distribution<int> dist(0, width);
		bodies[i].position = NBody::Vector2(dist(mt), dist(mt));
		double theta = rand() / double(RAND_MAX) * TAU;
		//bodies[i].velocity = sf::Vector2<double>(0, 0);
		bodies[i].velocity = NBody::Vector2(cos(theta), sin(theta)) * (rand() / double(RAND_MAX));
		if (i >= count - 3)
		{
			bodies[i].mass = SUN_MASS;
		}
		else
		{
			bodies[i].mass = (rand() / double(RAND_MAX)) * MASS;
		}
	}

	return bodies;
}

// BodyGenerator 개선 - 은하 형태 생성
std::vector<Body> BodyGenerator::generateSpiralBodies(int count, int width, int height)
{
	std::vector<Body> bodies(count);
	double centerX = width / 2.0;
	double centerY = height / 2.0;

	std::random_device rd;
	std::mt19937 gen(rd());
	static std::uniform_real_distribution<double> dist(0.0, 1.0);

	for (size_t i = 0; i < bodies.size(); ++i) {
		float a = dist(gen) * TAU;
		float sin_a = std::sin(a);
		float cos_a = std::cos(a);

		// 1. 거리 비율 생성 (0.0 ~ 1.0 사이, 중심에 밀집된 분포)
		float r_sum = 0.0f;
		for (int j = 0; j < 6; ++j) r_sum += dist(gen);

		// r은 확률적으로 0.0(중심) ~ 1.0(끝) 사이의 값을 가짐
		float r = std::abs(r_sum / 3.0f - 1.0f);
		float currentDist = r * height;

		// 위치 설정 (Local Space)
		NBody::Vector2 pos(cos_a * currentDist, sin_a * currentDist);
		NBody::Vector2 vel(sin_a, -cos_a);

		bodies[i].position = pos;
		bodies[i].velocity = vel;
		bodies[i].mass = MASS * dist(gen);
	}

	// 거리 정렬
	std::sort(bodies.begin(), bodies.end(), [](const Body& a, const Body& b) {
		return std::sqrt(a.position.magnitude()) < std::sqrt(b.position.magnitude());
	});

	// 궤도 속도 계산
	for (size_t i = 0; i < bodies.size(); ++i) {
		float d = std::sqrt(bodies[i].position.magnitude());
		if (d < 1e-6f) continue;

		float v = std::sqrt(static_cast<float>(i) / d);
		bodies[i].velocity *= v;
	}

	// 최종 위치 이동 (World Space)
	if (centerX != 0.0f || centerY != 0.0f) {
		for (auto& body : bodies) {
			body.position.x += centerX;
			body.position.y += centerY;
		}
	}
	return bodies;
}

std::vector<Body> BodyGenerator::generateSolarSystemBodies()
{
	std::vector<Body> bodies;
	// Sun
	Body sun;
	sun.position = NBody::Vector2(500.0, 500.0);
	sun.velocity = NBody::Vector2(0.0, 0.0);
	sun.mass = SUN_MASS;
	bodies.push_back(sun);
	// Earth
	Body earth;
	earth.position = NBody::Vector2(500.0 + 150.0, 500.0);
	earth.velocity = NBody::Vector2(0.0, 29780.0); // m/s
	earth.mass = 5.974e24;
	bodies.push_back(earth);
	// Mars
	Body mars;
	mars.position = NBody::Vector2(500.0 + 228.0, 500.0);
	mars.velocity = NBody::Vector2(0.0, 24070.0); // m/s
	mars.mass = 6.39e23;
	bodies.push_back(mars);
	// Venus
	Body venus;
	venus.position = NBody::Vector2(500.0 + 108.0, 500.0);
	venus.velocity = NBody::Vector2(0.0, 35020.0); // m/s
	venus.mass = 4.867e24;
	bodies.push_back(venus);

	return bodies;
}
