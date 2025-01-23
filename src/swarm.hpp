#ifndef _SWARM_HPP
#define _SWARM_HPP

#include "network.hpp"
#include <vector>
#include <iostream>

class Particle {
public:
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> pbest;

    Particle();

    void display();
};

class Swarm {
public:
    std::vector<Particle*> particles;
    std::vector<double> gbest;
    size_t num_particles;
    size_t num_dimensions;

    Swarm(size_t num_particles, Network n);

    void push_back(Particle* p);
    void display();
};

#endif