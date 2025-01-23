#ifndef _NETWORK_HPP
#define _NETWORK_HPP

#include <cstddef>
#include <vector>

class Network {
private:
    std::vector<int> nodes;
    std::vector<int> connections;
    size_t layers;
public:
    Network();
    Network(std::vector<int> nodes, std::vector<int> connections);

    std::vector<int> get_nodes();
    std::vector<int> get_connections();
    size_t get_size();
};

#endif