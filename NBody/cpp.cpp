#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>
#include <Windows.h>
#include <stack>
#include <omp.h>

#include "Body.h"
#include "DS_timer.h"

#define CPU

// CUDA
extern void CUDA_CleanUp();
extern void CUDA_Running(std::vector<Body>& bodies, double delta, int width, int height);
extern void CUDA_DataTransfer(std::vector<Body>& bodies);
// CUDA

float fps;
float deltaTime;
static double gpuComputeTime = 1;
static double gpuIntegrateTime;
static double gpuDataTransferTime;

const int width = 1000, height = 800;
const int worldMultiplier = 1000;
const int worldWidth = width * worldMultiplier;
const int worldHeight = height * worldMultiplier;

void SetTimes(double one, double two, double three)
{
	gpuComputeTime = one;
	gpuIntegrateTime = two;
	gpuDataTransferTime = three;
}

void UpdateFPS()
{
    static DWORD frameCount = 0;            //프레임 카운트수
    static float timeElapsed = 0.0f;            //흐른 시간
    static DWORD lastTime = timeGetTime();   //마지막 시간(temp변수)

    DWORD curTime = timeGetTime();      //현재 시간
    float timeDelta = (curTime - lastTime) * 0.001f;        //timeDelta(1번생성후 흐른 시간) 1초단위로 바꿔준다.

    timeElapsed += timeDelta;

    frameCount++;

    if (timeElapsed >= 1.0f)         //흐른시간이 1초이상이면 내가 하고싶은것 처리
    {
        fps = (float)frameCount / timeElapsed;

        frameCount = 0;
        timeElapsed = 0.0f;
    }

    lastTime = curTime;
    deltaTime = timeDelta;
}

struct Node 
{
	NBody::Vector2 position; // Center of mass
	NBody::Vector2 velocity;
	NBody::Vector2 acceleration;
	double mass; //total mass

    bool isInternal;
	double sizeX; // Size X of the quadrant
	double sizeY; // Size Y of the quadrant
	double centerX; // Center X of the quadrant
	double centerY; // Center Y of the quadrant

	Node() : position(0, 0), velocity(0, 0), acceleration(0, 0), mass(0), isInternal(false), sizeX(0), sizeY(0), centerX(0), centerY(0)
	{

	}
};

struct Tree
{
    Node root;
    Tree* children[4] = { nullptr, }; // NW, NE, SW, SE
};

Tree BarneshutTree;
#define NW 0
#define NE 1
#define SW 2
#define SE 3

int GetQuadrant(const Body& body, const Node& node)
{
    if (body.position.x < node.centerX && body.position.y < node.centerY)
    {
        return NW;
    }
    else if (body.position.x >= node.centerX && body.position.y < node.centerY)
    {
        return NE;
    }
    else if (body.position.x < node.centerX && body.position.y >= node.centerY)
    {
        return SW;
    }
    else
    {
        return SE;
    }
}

double GetCenterXQuadrant(int quadrant, const Node& parentNode)
{
	switch (quadrant)
	{
	case NW:
		return parentNode.centerX - parentNode.sizeX / 4;
	case NE:
		return parentNode.centerX + parentNode.sizeX / 4;
	case SW:
		return parentNode.centerX - parentNode.sizeX / 4;
	case SE:
		return parentNode.centerX + parentNode.sizeX / 4;
	default:
		return parentNode.centerX;
	}
}

double GetCenterYQuadrant(int quadrant, const Node& parentNode)
{
	switch (quadrant)
	{
	case NW:
		return parentNode.centerY - parentNode.sizeY / 4;
	case NE:
		return parentNode.centerY - parentNode.sizeY / 4;
	case SW:
		return parentNode.centerY + parentNode.sizeY / 4;
	case SE:
		return parentNode.centerY + parentNode.sizeY / 4;
	default:
		return parentNode.centerY;
	}
}

#define MIN_TREE_SIZE 1  // 최소 트리 크기를 정의
void InsertBodyToBarnesHutTree(const Body& body, Tree& parent)
{
    // 스택을 사용한 반복적 구현으로 변경
    std::stack<std::pair<const Body*, Tree*>> insertStack;
    insertStack.push({ &body, &parent });

    while (!insertStack.empty()) {
        auto [currentBody, currentTree] = insertStack.top();
        insertStack.pop();

        // 최소 크기 확인으로 무한 세분화 방지
        if (currentTree->root.sizeX < MIN_TREE_SIZE || currentTree->root.sizeY < MIN_TREE_SIZE) {
            // 크기가 너무 작으면 현재 노드에 질량 합계만 업데이트
            double totalMass = currentTree->root.mass + currentBody->mass;
            if (totalMass > 0) {
                currentTree->root.position = (currentTree->root.position * currentTree->root.mass +
                    currentBody->position * currentBody->mass) / totalMass;
                currentTree->root.mass = totalMass;
            }
            continue;
        }

        if (currentTree->root.isInternal) {
            int quadrant = GetQuadrant(*currentBody, currentTree->root);
            Tree* next = currentTree->children[quadrant];
            if (next == nullptr) {
                currentTree->children[quadrant] = new Tree();
                next = currentTree->children[quadrant];
                next->root.sizeX = currentTree->root.sizeX / 2;
                next->root.sizeY = currentTree->root.sizeY / 2;
                next->root.centerX = GetCenterXQuadrant(quadrant, currentTree->root);
                next->root.centerY = GetCenterYQuadrant(quadrant, currentTree->root);
            }

            insertStack.push({ currentBody, next });

            // 질량 중심 업데이트
            double totalMass = currentTree->root.mass + currentBody->mass;
            if (totalMass > 0) {
                currentTree->root.position = (currentTree->root.position * currentTree->root.mass +
                    currentBody->position * currentBody->mass) / totalMass;
                currentTree->root.mass = totalMass;
            }
        }
        else {
            if (currentTree->root.mass == 0) {
                // 빈 노드, 여기에 body 삽입
                currentTree->root.position = currentBody->position;
                currentTree->root.mass = currentBody->mass;
                currentTree->root.velocity = currentBody->velocity;
            }
            else {
                // 노드 세분화
                Body existingBody;
                existingBody.position = currentTree->root.position;
                existingBody.mass = currentTree->root.mass;
                existingBody.velocity = currentTree->root.velocity;

                currentTree->root.isInternal = true;

                // 자식 노드들 생성
                int existingQuadrant = GetQuadrant(existingBody, currentTree->root);
                int newQuadrant = GetQuadrant(*currentBody, currentTree->root);

                // 기존 body를 위한 자식 생성
                if (currentTree->children[existingQuadrant] == nullptr) {
                    currentTree->children[existingQuadrant] = new Tree();
                    currentTree->children[existingQuadrant]->root.sizeX = currentTree->root.sizeX / 2;
                    currentTree->children[existingQuadrant]->root.sizeY = currentTree->root.sizeY / 2;
                    currentTree->children[existingQuadrant]->root.centerX = GetCenterXQuadrant(existingQuadrant, currentTree->root);
                    currentTree->children[existingQuadrant]->root.centerY = GetCenterYQuadrant(existingQuadrant, currentTree->root);
                }

                // 새로운 body를 위한 자식 생성
                if (currentTree->children[newQuadrant] == nullptr) {
                    currentTree->children[newQuadrant] = new Tree();
                    currentTree->children[newQuadrant]->root.sizeX = currentTree->root.sizeX / 2;
                    currentTree->children[newQuadrant]->root.sizeY = currentTree->root.sizeY / 2;
                    currentTree->children[newQuadrant]->root.centerX = GetCenterXQuadrant(newQuadrant, currentTree->root);
                    currentTree->children[newQuadrant]->root.centerY = GetCenterYQuadrant(newQuadrant, currentTree->root);
                }

                // 스택에 삽입할 작업들 추가
                insertStack.push({ &existingBody, currentTree->children[existingQuadrant] });
                insertStack.push({ currentBody, currentTree->children[newQuadrant] });

                // 질량 중심 업데이트
                double totalMass = currentTree->root.mass + currentBody->mass;
                currentTree->root.position = (currentTree->root.position * currentTree->root.mass +
                    currentBody->position * currentBody->mass) / totalMass;
                currentTree->root.mass = totalMass;
            }
        }
    }
}

void ClearBarnesHutTree(Tree& tree)
{
    // 메모리 누수 방지를 위한 트리 정리 함수
    for (int i = 0; i < 4; i++) {
        if (tree.children[i] != nullptr) {
            ClearBarnesHutTree(*tree.children[i]);
            delete tree.children[i];
            tree.children[i] = nullptr;
        }
    }
    tree.root = Node(); // 노드 초기화
}

void buildBarnesHutTree(const std::vector<Body>& bodies)
{
    // 기존 트리 정리
    ClearBarnesHutTree(BarneshutTree);

    // QuadTree
	BarneshutTree.root.sizeX = worldWidth;
	BarneshutTree.root.sizeY = worldHeight;
	BarneshutTree.root.centerX = worldWidth / 2;
	BarneshutTree.root.centerY = worldHeight / 2;
	BarneshutTree.root.isInternal = false;
    
	for (int i = 0; i < bodies.size(); i++)
	{
		InsertBodyToBarnesHutTree(bodies[i], BarneshutTree);
	}
}

void computeForces(std::vector<Body>& bodies) {
    int N = bodies.size();

    for (int i = 0; i < N; i++) {
        NBody::Vector2 force(0.f, 0.f);

        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            NBody::Vector2 r = bodies[j].position - bodies[i].position;
            double distSqr = r.x * r.x + r.y * r.y + EPS * EPS;
			double distSixth = distSqr * distSqr * distSqr;
            double invDist = 1.0f / std::sqrt(distSixth);

            double f = G * bodies[i].mass * bodies[j].mass * invDist;
            force += r * f;
        }

        bodies[i].acceleration = force; // F = ma (m_i 약분됨)
    }
}

void computeForcesBarnesHut(std::vector<Body>& bodies) {
    int N = bodies.size();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        NBody::Vector2 force(0.f, 0.f);
        // Barnes-Hut 알고리즘을 사용하여 힘 계산
        std::stack<Tree*> nodeStack;
        nodeStack.push(&BarneshutTree);
        while (!nodeStack.empty()) {
            Tree* currentNode = nodeStack.top();
            nodeStack.pop();
            if (currentNode == nullptr || currentNode->root.mass == 0) continue;
            NBody::Vector2 r = currentNode->root.position - bodies[i].position;
            double distSqr = r.x * r.x + r.y * r.y + EPS * EPS;
            double dist = std::sqrt(distSqr);
            // 트리 노드의 크기와 거리 비율 계산
            double s = currentNode->root.sizeX; // 노드의 크기 (가로 길이)
            if ((s / dist) < THETA) {
                // 멀리 있는 노드로 간주, 질량 중심으로부터 힘 계산
                double invDist = 1.0 / (distSqr * dist);
                double f = G * currentNode->root.mass * invDist;
                force += r * f;
            }
            else {
                // 노드가 너무 가까우므로 자식 노드들을 스택에 추가
                for (int k = 0; k < 4; k++) {
                    if (currentNode->children[k] != nullptr) {
                        nodeStack.push(currentNode->children[k]);
                    }
                }
            }
        }
        bodies[i].acceleration = force; // F = ma (m_i 약분됨)
    }
}

void integrate(std::vector<Body>& bodies) {
    #pragma omp parallel for
    for (int i = 0; i < bodies.size(); ++i) 
    {
        bodies[i].velocity += bodies[i].acceleration * deltaTime * DT;
        bodies[i].position += bodies[i].velocity * deltaTime * DT;

        if (bodies[i].position.x < 0 || bodies[i].position.x > worldWidth) bodies[i].velocity.x *= -0.9f;
        if (bodies[i].position.y < 0 || bodies[i].position.y > worldHeight) bodies[i].velocity.y *= -0.9f;

        bodies[i].position.x = std::clamp(bodies[i].position.x, 0., (double)worldWidth);
        bodies[i].position.y = std::clamp(bodies[i].position.y, 0., (double)worldHeight);
    }
}

const char* FPS_TEXT = "CPU N-Body Simulation - FPS : ";
const char* BODY_COUNT_TEXT = "Body Count : ";
const char* DRAWTIME_TEXT = "Draw Time : ";
const char* GPUCOMPUTE_TEXT = "GPU Compute Time : ";
const char* GPUINTEGRATE_TEXT = "GPU Integrate Time : ";
const char* GPUTRANSFER_TEXT = "GPU Data Transfer Time : ";

sf::Vector2f WorldToScreenPos(const double x, const double y)
{
    float screenX = (x / worldWidth) * (width - 5);
    float screenY = (y / worldHeight) * (height - 5);
    return sf::Vector2f(screenX, screenY);
}

sf::Vector2f WorldToScreenPosNotOffset(const double x, const double y)
{
    float screenX = (x / worldWidth) * (width);
    float screenY = (y / worldHeight) * (height);
    return sf::Vector2f(screenX, screenY);
}

void DrawBoundary(sf::RenderWindow& window, sf::RectangleShape rect, const Tree& node)
{
	std::stack<const Tree*> nodeStack;
	nodeStack.push(&node);

    while (!nodeStack.empty())
    {
		const Tree* currentNode = nodeStack.top();
		nodeStack.pop();
		if (currentNode != nullptr && currentNode->root.isInternal)
		{
			rect.setSize(sf::Vector2f(currentNode->root.sizeX / worldMultiplier, currentNode->root.sizeY / worldMultiplier));
			rect.setPosition(sf::Vector2f(WorldToScreenPosNotOffset(currentNode->root.centerX - currentNode->root.sizeX / 2, currentNode->root.centerY - currentNode->root.sizeY / 2)));
			window.draw(rect);
			// Push children to stack
			nodeStack.push(currentNode->children[NW]);
			nodeStack.push(currentNode->children[NE]);
			nodeStack.push(currentNode->children[SW]);
			nodeStack.push(currentNode->children[SE]);
		}
    }
}

sf::Color Lerp(sf::Color A, sf::Color B, float Alpha)
{
    sf::Color newColor;
	newColor.r = A.r * (1 - Alpha) + B.r * Alpha;
	newColor.g = A.g * (1 - Alpha) + B.g * Alpha;
	newColor.b = A.b * (1 - Alpha) + B.b * Alpha;
	return newColor;
}

bool isDrawingQuadTree = false;
bool F1KeyPressedLastFrame = false;

int main() {
    std::string title = "CPU N-Body Simulation - FPS : 0";
    sf::RenderWindow window(sf::VideoMode({ width, height }), title);
    window.setFramerateLimit(144);

    DS_timer timer(2);
    
    // TEXT
    sf::Font font("Pretendard-SemiBold.ttf");
    sf::Text bodyCount(font);
	bodyCount.setCharacterSize(20);
	bodyCount.setString("Body Count : " + std::to_string(N));
	bodyCount.setFillColor(sf::Color::White);
	bodyCount.setPosition(sf::Vector2f(0, 0));

    sf::Text drawTime(font);
    drawTime.setCharacterSize(20);
    drawTime.setFillColor(sf::Color::White);
    drawTime.setPosition(sf::Vector2f(0, 20));

    sf::Text gpuComputeTimeText(font);
    gpuComputeTimeText.setCharacterSize(20);
    gpuComputeTimeText.setFillColor(sf::Color::White);
    gpuComputeTimeText.setPosition(sf::Vector2f(0, 40));

    sf::Text gpuIntegrateTimeText(font);
    gpuIntegrateTimeText.setCharacterSize(20);
    gpuIntegrateTimeText.setFillColor(sf::Color::White);
    gpuIntegrateTimeText.setPosition(sf::Vector2f(0, 60));

    sf::Text gpuTransferTimeText(font);
    gpuTransferTimeText.setCharacterSize(20);
    gpuTransferTimeText.setFillColor(sf::Color::White);
    gpuTransferTimeText.setPosition(sf::Vector2f(0, 80));

    std::vector<Body> bodies;

	//bodies = BodyGenerator::generateRandomBodies(N, worldWidth, worldHeight);
	bodies = BodyGenerator::generateSpiralBodies(N, worldWidth, worldHeight);
	//bodies = BodyGenerator::generateSolarSystemBodies();

    CUDA_DataTransfer(bodies);

    // QuadTree
	buildBarnesHutTree(bodies);
	sf::RectangleShape boundary(sf::Vector2f(width, height));
	boundary.setPosition(sf::Vector2f(0, 0));
	boundary.setFillColor(sf::Color::Transparent);
	boundary.setOutlineColor(sf::Color::Red);
	boundary.setOutlineThickness(0.75f);

    // Particle
    sf::CircleShape particle(0.6f);
    sf::CircleShape particleSun(5);
    particleSun.setFillColor(sf::Color::Yellow);

    // 시뮬레이션 루프
    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }
        UpdateFPS();

        // CPU
#ifdef CPU
        //computeForces(bodies);
		timer.initTimer(1);
		timer.onTimer(1);
		computeForcesBarnesHut(bodies);
        integrate(bodies);
        buildBarnesHutTree(bodies);
		timer.offTimer(1);
		gpuComputeTime = timer.getTimer_s(1);
#else
        // GPU
        CUDA_Running(bodies, deltaTime, worldWidth, worldHeight);
#endif

		timer.initTimer(0);
		timer.onTimer(0);
        window.setTitle(FPS_TEXT + std::to_string(fps));
        window.clear(sf::Color::Black);

		// Draw Bodies
        for (int i = 0; i < bodies.size(); ++i) {
            if (bodies[i].mass >= SUN_MASS)
            {
                particleSun.setPosition(WorldToScreenPos(bodies[i].position.x, bodies[i].position.y));
                window.draw(particleSun);
            }
            else
            {
                float lerp = std::clamp(bodies[i].velocity.magnitude() * 2, 0., 1.);
                particle.setFillColor(Lerp(sf::Color::White, sf::Color::Green, lerp));
                particle.setPosition(WorldToScreenPos(bodies[i].position.x, bodies[i].position.y));
                particle.setRadius((bodies[i].mass / MASS) * 0.9f + 0.6f);
                window.draw(particle);
            }
        }


		// Draw QuadTree
		bool F1KeyPressedThisFrame = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::F1);
        if (F1KeyPressedThisFrame && !F1KeyPressedLastFrame)
        {
			isDrawingQuadTree = !isDrawingQuadTree;
        }
        F1KeyPressedLastFrame = F1KeyPressedThisFrame;

		if (isDrawingQuadTree)
		    DrawBoundary(window, boundary, BarneshutTree);

        //TEXT
        window.draw(bodyCount);
		gpuComputeTimeText.setString(GPUCOMPUTE_TEXT + std::to_string(gpuComputeTime));
		gpuIntegrateTimeText.setString(GPUINTEGRATE_TEXT+ std::to_string(gpuIntegrateTime));
		gpuTransferTimeText.setString(GPUTRANSFER_TEXT + std::to_string(gpuDataTransferTime));
		window.draw(gpuComputeTimeText);
		window.draw(gpuIntegrateTimeText);
		window.draw(gpuTransferTimeText);
        
        timer.offTimer(0);
        drawTime.setString(DRAWTIME_TEXT + std::to_string(timer.getTimer_s(0)));
		window.draw(drawTime);

        window.display();
    }

    CUDA_CleanUp();

    return 0;
}
