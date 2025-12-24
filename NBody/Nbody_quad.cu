#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector.h"
#include "DS_timer.h"

#include <vector>

template<typename T>
struct DeviceBuffer {
private:
	T* ptr = nullptr;
	const char* name = nullptr;

public:
	DeviceBuffer(const char* name) : name(name) {}

	T* get() { return ptr; }

	void alloc(size_t len) {
		cudaMalloc(&ptr, len * sizeof(T));
	}

	void free() {
		cudaFree(ptr);

		ptr = nullptr;
	}
};

static double gpuComputeTime;
static double gpuIntegrateTime;
static double gpuDataTransferTime;

extern void SetTimes(double one, double two, double three);

struct Body {
	NBody::Vector2 position;
	NBody::Vector2 velocity;
	NBody::Vector2 acceleration;
	double mass;
};

double* d_posX2;
double* d_posY2;
double* d_velX2;
double* d_velY2;
double* d_accX2;
double* d_accY2;
double* d_mass2;

double* srcPosX2;
double* srcPosY2;
double* srcVelX2;
double* srcVelY2;
double* srcAccX2;
double* srcAccY2;
double* srcMass2;

int NUM_BODIES;

__host__ void CUDA_DataTransfer_quad(std::vector<Body>& bodies)
{
	NUM_BODIES = bodies.size();

	cudaMalloc(&d_posX, sizeof(double) * NUM_BODIES);
	cudaMalloc(&d_posY, sizeof(double) * NUM_BODIES);
	cudaMalloc(&d_velX, sizeof(double) * NUM_BODIES);
	cudaMalloc(&d_velY, sizeof(double) * NUM_BODIES);
	cudaMalloc(&d_accX, sizeof(double) * NUM_BODIES);
	cudaMalloc(&d_accY, sizeof(double) * NUM_BODIES);
	cudaMalloc(&d_mass, sizeof(double) * NUM_BODIES);

	srcPosX = new double[NUM_BODIES];
	for (size_t i = 0; i < NUM_BODIES; i++)
	{
		srcPosX[i] = bodies[i].position.x;
	}
	srcPosY = new double[NUM_BODIES];
	for (size_t i = 0; i < NUM_BODIES; i++)
	{
		srcPosY[i] = bodies[i].position.y;
	}
	srcVelX = new double[NUM_BODIES];
	for (size_t i = 0; i < NUM_BODIES; i++)
	{
		srcVelX[i] = bodies[i].velocity.x;
	}
	srcVelY = new double[NUM_BODIES];
	for (size_t i = 0; i < NUM_BODIES; i++)
	{
		srcVelY[i] = bodies[i].velocity.y;
	}
	srcAccX = new double[NUM_BODIES];
	for (size_t i = 0; i < NUM_BODIES; i++)
	{
		srcAccX[i] = bodies[i].acceleration.x;
	}
	srcAccY = new double[NUM_BODIES];
	for (size_t i = 0; i < NUM_BODIES; i++)
	{
		srcAccY[i] = bodies[i].acceleration.y;
	}
	srcMass = new double[NUM_BODIES];
	for (size_t i = 0; i < NUM_BODIES; i++)
	{
		srcMass[i] = bodies[i].mass;
	}

	cudaMemcpy(d_posX, srcPosX, sizeof(double) * NUM_BODIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_posY, srcPosY, sizeof(double) * NUM_BODIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velX, srcVelX, sizeof(double) * NUM_BODIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velY, srcVelY, sizeof(double) * NUM_BODIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accX, srcAccX, sizeof(double) * NUM_BODIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accY, srcAccY, sizeof(double) * NUM_BODIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, srcMass, sizeof(double) * NUM_BODIES, cudaMemcpyHostToDevice);
}

__global__ void Integrate_quad(double* dX, double* dY, double* dVelX, double* dVelY, double* dAccX, double* dAccY, double DT, int NUM_BODIES, int width, int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= NUM_BODIES) return;

	dVelX[idx] += dAccX[idx] * DT;
	dVelY[idx] += dAccY[idx] * DT;

	dX[idx] += dVelX[idx] * DT;
	dY[idx] += dVelY[idx] * DT;

	if (dX[idx] < 0 || dX[idx] > width) dVelX[idx] *= -0.9f;
	if (dY[idx] < 0 || dY[idx] > height) dVelY[idx] *= -0.9f;

	dX[idx] = dX[idx] < 0 ? 0 : (dX[idx] > (double)width ? (double)width : dX[idx]);
	dY[idx] = dY[idx] < 0 ? 0 : (dY[idx] > (double)height ? (double)height : dY[idx]);
}

__global__ void ComputeForces_quad(double* dX, double* dY, double* dVelX, double* dVelY, double* dAccX, double* dAccY, double* dMass, int NUM_BODIES, double EPS, double G)
{
	double forceX = 0;
	double forceY = 0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= NUM_BODIES) return;

	for (int i = 0; i < NUM_BODIES; ++i) {
		if (i == idx) continue;

		double rx = dX[i] - dX[idx];
		double ry = dY[i] - dY[idx];
		double distSqr = rx * rx + ry * ry + EPS * EPS;
		double distSixth = distSqr * distSqr * distSqr;
		double invDist = 1.0f / sqrt(distSixth);

		double f = G * dMass[i] * invDist;
		forceX += rx * f;
		forceY += ry * f;
	}

	dAccX[idx] = forceX;
	dAccY[idx] = forceY;
}

__host__ void CUDA_Running_quad(std::vector<Body>& bodies, double delta, int width, int height)
{
	DS_timer timer(3);

	dim3 BLOCK_SIZE(32);
	dim3 GRID_SIZE(ceil((float)NUM_BODIES / BLOCK_SIZE.x));

	timer.initTimer(0);
	timer.onTimer(0);
	ComputeForces_quad << < GRID_SIZE, BLOCK_SIZE >> > (d_posX, d_posY, d_velX, d_velY, d_accX, d_accY, d_mass, NUM_BODIES, EPS, G);
	cudaDeviceSynchronize();
	timer.offTimer(0);
	gpuComputeTime = timer.getTimer_s(0);

	timer.initTimer(1);
	timer.onTimer(1);
	Integrate_quad << < GRID_SIZE, BLOCK_SIZE >> > (d_posX, d_posY, d_velX, d_velY, d_accX, d_accY, delta * DT, NUM_BODIES, width, height);
	cudaDeviceSynchronize();
	timer.offTimer(1);
	gpuIntegrateTime = timer.getTimer_s(1);

	timer.initTimer(2);
	timer.onTimer(2);
	cudaMemcpy(srcPosX, d_posX, sizeof(double) * NUM_BODIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(srcPosY, d_posY, sizeof(double) * NUM_BODIES, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < NUM_BODIES; i++)
	{
		bodies[i].position.x = srcPosX[i];
		bodies[i].position.y = srcPosY[i];
	}
	timer.offTimer(2);
	gpuDataTransferTime = timer.getTimer_s(2);
	SetTimes(gpuComputeTime, gpuIntegrateTime, gpuDataTransferTime);
}

__host__ void CUDA_CleanUp_quad()
{
	cudaFree(d_posX);
	cudaFree(d_posY);
	cudaFree(d_velX);
	cudaFree(d_velY);
	cudaFree(d_accX);
	cudaFree(d_accY);
	cudaFree(d_mass);

	delete[] srcPosX;
	delete[] srcPosY;
	delete[] srcVelX;
	delete[] srcVelY;
	delete[] srcMass;
}