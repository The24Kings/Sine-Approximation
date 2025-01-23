#include "network.hpp"

Network::Network() {
    this->nodes = std::vector<int>();
    this->connections = std::vector<int>();
    this->layers = 0;
}

Network::Network(std::vector<int> nodes, std::vector<int> connections) {
    this->nodes = nodes;
    this->connections = connections;
    this->layers = connections.size();
}

std::vector<int> Network::get_connections() {
    return connections;
}

std::vector<int> Network::get_nodes() {
    return nodes;
}

size_t Network::get_size() {
    return layers;
}