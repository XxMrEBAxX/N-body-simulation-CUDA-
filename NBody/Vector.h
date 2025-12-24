#pragma once

const double G = 6.67E-7;     // 중력 상수 (시뮬레이션 스케일에 맞게 조정)
const double DT = 50000;    // 시간 간격 (time step)
const double EPS = 5000;   // softening (0으로 하면 불안정할 수 있음)
const double MASS = 10e6; //5.974e24;
const double SUN_MASS = 10e12; // 태양 질량
const int N = 10000; // 입자 수
const double THETA = 0.75; // 임계값

namespace NBody
{
	struct Vector2
	{
		double x;
		double y;

		Vector2() : x(0), y(0) {}
		Vector2(double x, double y) : x(x), y(y) {}

		Vector2 operator+(const Vector2& other) const
		{
			return { x + other.x, y + other.y };
		}

		Vector2 operator-(const Vector2& other) const
		{
			return { x - other.x, y - other.y };
		}

		Vector2 operator*(double scalar) const
		{
			return { x * scalar, y * scalar };
		}

		Vector2 operator/(double scalar) const
		{
			return { x / scalar, y / scalar };
		}

		Vector2& operator+=(const Vector2& other)
		{
			x += other.x;
			y += other.y;
			return *this;
		}

		Vector2& operator-=(const Vector2& other)
		{
			x -= other.x;
			y -= other.y;
			return *this;
		}

		Vector2& operator*=(double scalar)
		{
			x *= scalar;
			y *= scalar;
			return *this;
		}

		Vector2& operator/=(double scalar)
		{
			x /= scalar;
			y /= scalar;
			return *this;
		}

		double magnitude() const
		{
			return x * x + y * y;
		}
	};
}


