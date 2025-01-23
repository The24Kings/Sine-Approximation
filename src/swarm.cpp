#include "swarm.hpp"
#include "network.hpp"

Swarm::Swarm(size_t num_particles, Network n) {
    size_t layers = n.get_size();
    std::vector<int> connections = n.get_connections();

    // Size of the swarm is determined by the size of the network
    int size = 0;

    for (size_t i = 0; i < layers; i++) {
        size += connections[i];
    }

    this->num_particles = num_particles;
    this->num_dimensions = size;
    particles = std::vector<Particle*>();
}

void Swarm::push_back(Particle* p) {
    particles.push_back(p);
}

/**
 * @brief Displays the swarm
 * 
 */
void Swarm::display() {
    for (size_t i = 0; i < num_particles; i++) {
        std::cout << "Particle " << i << ":\n";
        std::cout << "\tPosition: ";
        for (size_t j = 0; j < num_dimensions; j++) {
            std::cout << particles[i]->position[j] << " ";
        }
        std::cout << "\n";

        std::cout << "\tVelocity: ";
        for (size_t j = 0; j < num_dimensions; j++) {
            std::cout << particles[i]->velocity[j] << " ";
        }
        std::cout << "\n";

        std::cout << "\tPersonal Best: ";
        for (size_t j = 0; j < num_dimensions; j++) {
            std::cout << particles[i]->pbest[j] << " ";
        }
        std::cout << "\n\n";
    }
}

Particle::Particle() {
    position = std::vector<double>();
    velocity = std::vector<double>();
    pbest = std::vector<double>();
}

void Particle::display() {
    std::cout << "Position: ";
    for (size_t i = 0; i < position.size(); i++) {
        std::cout << position[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Velocity: ";
    for (size_t i = 0; i < velocity.size(); i++) {
        std::cout << velocity[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Personal Best: ";
    for (size_t i = 0; i < pbest.size(); i++) {
        std::cout << pbest[i] << " ";
    }
    std::cout << "\n\n";
}