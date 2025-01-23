#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <random>
#include <sys/stat.h>
#include <csignal>
#include <ctime>
#include <chrono>
#include <limits>
#include <cmath>
#include <algorithm>
#include <set>
#include <stdarg.h>
#include <matplot/matplot.h>

#define RAND(MIN, MAX) MIN + static_cast<double> (rand()) /( static_cast<double> (RAND_MAX/(MAX-MIN)))
#define M_PI 3.14159265358979323846
//#define _DEBUG

#include "network.hpp"
#include "swarm.hpp"
#include "matrix.hpp"

enum LOG_LEVEL {
    INFO,
    WARN
};

/* PSO Parameters */

double intertia = 0.99; // Inertia weight (w)
double cognitive = 2.0; // Cognitive weight (c1)
double social = 2.0; // Social weight (c2)
double r1, r2; // Random numbers - To avoid local minima   

int epochs = 1000;
int max_init_pos = 20;
int min_init_pos = -20;

int max_init_vel = 20;
int min_init_vel = -20;

/* Global Variables */

int num_particles = 30;
double step = 0.05;
double min_sin = -3 * M_PI;
double max_sin = 3 * M_PI;

/* Utility Functions */

Matrix<double> columnwise_sum(Matrix<double> x);
double columnwise_sum(std::vector<double> x);
Matrix<double>& transpose(Matrix<double> &m);
Matrix<double> generate_sin_data(double start, double end, double step);

/* Neural Network Functions */

Matrix<double> BiPS_activate(Matrix<double> &x);
Matrix<double> weight_matrix(std::vector<double> pos, Network* n, size_t layer);
double neural_output(Network* n, std::vector<double> p, double i);

/* PSO Functions */

double update_velocity(double w, double v, double c1, double c2, double pos, double r1, double r2, double pbestPos, double gbestPos);
double update_position(double pos, double vel);
double RB_fitness(std::vector<double> pos, Matrix<double> &expected, Network &network);
double MSE_fitness(std::vector<double> pos, Matrix<double> &expected, Network &network);

/* Globals for sigact */

Matrix<double> sin_data;
Matrix<double> actual_sin;
Matrix<double> predicted_sin;
size_t iterations;
Network global_network;
Swarm* swarm;

int main() {
    srand(time(0));
    std::cout << std::fixed << std::setprecision(2);

    // Generate the sin wave data
    sin_data = generate_sin_data(min_sin, max_sin, 0.05); // Expected sin wave data
    actual_sin = Matrix<double>(1, sin_data.row); // Actual sin wave output
    predicted_sin = Matrix<double>(1, sin_data.row); // Predicted sin wave output

    // Set actual sin data
    for (size_t i = 0; i < sin_data.row; i++) {
        actual_sin[i][0] = sin_data[i][1];
    }

    // Number of connections in the network per layer
    std::vector<int> nodes = {1, 5, 5, 1};
    std::vector<int> connections = {1, 5, 25, 5, 1};

    global_network = Network(nodes, connections);
    swarm = new Swarm(num_particles, global_network);

    // Initialize the swarm
    for (size_t i = 0; i < swarm->num_particles; i++) {
        Particle* p = new Particle();
        swarm->push_back(p);

        // Initialize the position
        for (size_t j = 0; j < swarm->num_dimensions; j++) {
            p->position.push_back(RAND(min_init_pos, max_init_pos));
        }

        // Initialize the velocity
        for (size_t j = 0; j < swarm->num_dimensions; j++) {
            p->velocity.push_back(RAND(min_init_vel, max_init_vel));
        }

        // Initialize the personal best
        p->pbest = p->position;
    }

    // Initialzise the global best
    swarm->gbest = swarm->particles[0]->position;

    // Initialize the Figure
    auto h = matplot::figure();
    h->size(1280, 720);

    // Particle Swarm Optimization 
    r1 = RAND(0, 1);
    r2 = RAND(0, 1);

    double currentFitness = MSE_fitness(swarm->particles[0]->position, actual_sin, global_network);
    double pbestFitness = currentFitness;
    double gbestFitness = currentFitness;

    for (iterations = 0; iterations < epochs; iterations++) {
        printf("Epoch: %zu/%d - Current Best: %.4f\n", iterations, epochs, gbestFitness);

        //if (gbestFitness < 1.01) { break; } // Break if the fitness is less than 1.01 (we have reached the global minimum)

        for (size_t j = 0; j < swarm->num_particles; j++) {
            Particle* current = swarm->particles[j];

            // Update the velocity
            for (size_t k = 0; k < swarm->num_dimensions; k++) {
                current->velocity[k] = update_velocity(
                    intertia, 
                    current->velocity[k], 
                    cognitive, 
                    social, 
                    current->position[k], 
                    r1, 
                    r2, 
                    current->pbest[k], 
                    swarm->gbest[k]
                );
            }

            // Update the position
            for (size_t k = 0; k < swarm->num_dimensions; k++) {
                current->position[k] = update_position(current->position[k], current->velocity[k]);
            }

            // Calculate the fitness
            currentFitness = MSE_fitness(current->position, actual_sin, global_network);

            // Update the personal best
            if (currentFitness < pbestFitness) {
                current->pbest = current->position;
                pbestFitness = currentFitness;
            }

            // Update the global best
            if (currentFitness < gbestFitness) {
                swarm->gbest = current->position;
                gbestFitness = currentFitness;

                // Predicted sin wave data
                std::vector<double> x;
                std::vector<double> y;

                // Actual sin wave data
                std::vector<double> x2;
                std::vector<double> y2;

                for (size_t i = 0; i < sin_data.row; i++) {
                    x.push_back(sin_data[i][0]);
                    x2.push_back(sin_data[i][0]);
                    y.push_back(neural_output(&global_network, swarm->gbest, sin_data[i][1]));
                    y2.push_back(sin_data[i][1]);
                }
            
                // Velocity
                std::vector<double> x3;
                std::vector<double> y3;

                x3 = matplot::linspace(-10, 10, current->velocity.size());

                for (size_t i = 0; i < current->velocity.size(); i++) {
                    y3.push_back(current->velocity[i]);
                }

                // Predicted sin wave data
                matplot::plot(x, y, "r*-", x2, y2, "bs-", x3, y3, "ko-");
                matplot::yrange({-1.5, 1.5});

                //h->draw();
                matplot::show(); // In order for this to be non-blocking you need to remove the matplot::wait(); from backend_interface::show() 
            }
        }
    }

    // Print out the output of the glboalbest 
    // From -3pi to 3pi input into the neural network and get the output
    std::cout << "\nFinal fitness: " << gbestFitness << std::endl;

    std::cout << "Sin : Actual : Predicted" << std::endl;
    for (size_t i = 0; i < actual_sin.row; i++) {
        std::cout << sin_data[i][0] << " : " << sin_data[i][1] << " : " << neural_output(&global_network, swarm->gbest, sin_data[i][1]) << std::endl;
    }

    // Show the weights
    std::cout << "\nWeights: ";
    for (size_t i = 0; i < global_network.get_size(); i++) {
        std::cout << "\nLayer: " << i << std::endl;
        weight_matrix(swarm->gbest, &global_network, i).print_matrix();
    }

    matplot::wait();

    return 0;
}  

/**
 * @brief Sum up each column in a Matrix
 * 
 * @param x Input Matrix
 * @return Matrix<double>
 */
Matrix<double> columnwise_sum(Matrix<double> x) {
    Matrix<double> sum(1, x.row);

    for (size_t i = 0; i < x.row; i++) {
        for (size_t j = 0; j < x.col; j++) {
            sum[i][j] += x[i][j];
        }
    }

    return sum;
}

/**
 * @brief Sum up each index in a vector
 * 
 * @param x Input vector
 * @return double 
 */
double columnwise_sum(std::vector<double> x) {
    double sum = 0;

    for (size_t i = 0; i < x.size(); i++) {
        sum += x[i];
    }

    return sum;
}

/**
 * @brief Rotates a matrix by 90 degrees
 * 
 * @param m Input matrix
 * @return Matrix 
 */
Matrix<double>& transpose(Matrix<double> &m) {
    static Matrix<double> t(m.row, m.col);

    for (size_t i = 0; i < m.row; i++) {
        for (size_t j = 0; j < m.col; j++) {
            t[j][i] = m[i][j];
        }
    }

    return t;
}

/**
 * @brief Calculates the fitness of a particle based on the error between the predicted and actual outputs
 * 
 * @param pos Particle Position to calculate the fitness for
 * @param expected Expected output
 * @param network Neural network
 * 
 * @return double 
 */
double RB_fitness(std::vector<double> pos, Matrix<double> &expected, Network &network) {
    Matrix<double> predicted(1, expected.row);
    double sum = 0;

    // From -3pi to 3pi input into the neural network and get the output
    for (size_t i = 0; i < expected.row; i++) {
        predicted[i][0] = neural_output(&network, pos, expected[i][0]);
    }

    // Calculate the error
    Matrix<double> error = expected - predicted;
    Matrix<double>& eT = transpose(error); // Essentially a std::vector<double>

    // Calculate the fitness using the Rosenbrock function
    for (size_t i = 0; i < eT.col; i++) {
        sum += 100 * std::pow(eT[0][i + 1] - std::pow(eT[0][i], 2), 2) + std::pow(1 - eT[0][i], 2);
    }

    return sum;
}

/**
 * @brief Calculates the fitness of a particle based on the error between the predicted and actual outputs
 * 
 * @param pos Particle Position to calculate the fitness for
 * @param expected Expected output
 * @param network Neural network
 * 
 * @return double 
 */
double MSE_fitness(std::vector<double> pos, Matrix<double> &expected, Network &network) {
    Matrix<double> predicted(1, expected.row);
    double sum = 0;

    // From -3pi to 3pi input into the neural network and get the output
    for (size_t i = 0; i < expected.row; i++) {
        predicted[i][0] = neural_output(&network, pos, expected[i][0]);
    }

    // Calculate the error MSE
    for (size_t i = 0; i < expected.row; i++) {
        sum += std::pow(expected[i][0] - predicted[i][0], 2);
    }

    return (1.0 / expected.row) * sum;
}

/**
 * @brief Generates the sin data for the given range
 * 
 * @param start Start value
 * @param end End value
 * @param step Step value
 * @return Matrix<double> 
 */
Matrix<double> generate_sin_data(double start, double end, double step) {
    size_t rows = (end - start) / step;
    Matrix<double> data(2, rows);

    for (size_t i = 0; i < rows; i++) {
        data[i][0] = start + step * i;
        data[i][1] = sin(data[i][0]);
    }

    return data;
}

/**
 * @brief Applies the bipolar sigmoid activation function to a matrix
 * 
 * @param x Weighted Input Matrix
 * 
 * @return Matrix<double> 
 */
Matrix<double> BiPS_activate(Matrix<double> &x) {
    // Initialize the output matrix
    Matrix<double> out(x.col, x.row);

    // Use the bipolar sigmoid function as the activation function
    for (size_t i = 0; i < out.row; i++) {
        for (size_t j = 0; j < out.col; j++) {
            double val = x[i][j];
            double denom = 1.0 + exp(-val);

            out[i][j] = -1.0 + (2.0 / denom); // Bipolar sigmoid
        }
    }

    return out;
}

/**
 * @brief Updates the velocity of a particle
 * 
 * @param w Inertia weight
 * @param v Current velocity
 * @param c1 Cognitive coefficient
 * @param c2 Social coefficient
 * @param pos Current position
 * @param r1 Random number 1
 * @param r2 Random number 2
 * @param pbestPos Personal best position
 * @param gbestPos Global best position
 * @return double 
 */
double update_velocity(double w, double v, double c1, double c2, double pos, double r1, double r2, double pbestPos, double gbestPos) {
    double vel = 0;

    vel = w * v + c1 * r1 * (pbestPos - pos) + c2 * r2 * (gbestPos - pos);

    // Check if the velocity is within the bounds
    if (vel > max_init_vel) {
        return max_init_vel;
    } else if (vel < min_init_vel) {
        return min_init_vel;
    } else {
        return vel;
    }
}

/**
 * @brief Updates the position of a particle
 * 
 * @param pos Current position
 * @param vel Updated velocity
 * @return double 
 */
double update_position(double cpos, double vel) {
    double pos = 0;

    pos = cpos + vel;

    // Check if the position is within the bounds (-1000, 1000)
    if (pos > max_init_pos) {
        return max_init_pos;
    } else if (pos < min_init_pos) {
        return min_init_pos;
    } else {
        return pos;
    }
}

/**
 * @brief Get the weight matrix for a given layer of the network (turns a 1D array into a 2D matrix)
 * 
 * @param p Particle Position
 * @param n Network layout
 * @param conn_layer Connection layer
 * @return Matrix<double> 
 */
Matrix<double> weight_matrix(std::vector<double> pos, Network* n, size_t conn_layer) {
    int start = 0;
    Matrix<double> weights;
    std::vector<int> nodes = n->get_nodes();
    std::vector<int> connections = n->get_connections();

    // Find the starting index for the current layer
    for (size_t i = 0; i < conn_layer; i++) {
        start += connections[i];
    }

    // Find the dimensions of the weight matrix
    int x = (conn_layer > nodes.size() - 1) ? 1 : nodes[conn_layer]; // If the layer is greater than the list of nodes, set the width to 1
    int y = 0;

    // Get the number of columns
    for (size_t i = 0; i < connections[conn_layer]; i++) {
        if (i % x == 0) {
            y++;
        }
    }

    // Initialize the weight matrix
    weights = Matrix<double>(x, y);

    // Initialize the weights
    for (size_t i = 0; i < weights.row; i++) {
        for (size_t j = 0; j < weights.col; j++) {
            weights[i][j] = pos[start + j + (x * i)]; // The starting index is determined by the sum of connections in the previous layers
        }
    }

    return weights;
}

/**
 * @brief Get the output of the neural network for a given input
 * 
 * @param n Network
 * @param p Particle containing the weights
 * @param i Input
 * @return double 
 */
double neural_output(Network* n, std::vector<double> p, double i) {
    size_t layers = n->get_size();
    Matrix<double> input = i; // Input should be a 1x1 matrix to start with
    Matrix<double> output; 
    Matrix<double> weights;

    // For each layer in the network
    for (size_t j = 0; j < layers; j++) {
        // Get the weights for the current layer
        weights = weight_matrix(p, n, j);

        // Multiply the input by the weights
        output = input * weights; 

        // Apply the activation function
        output = BiPS_activate(output);

        // Set the input to the output for the next layer
        input = output;
    }

    return output[0][0]; // Output should be a 1x1 matrix with the final output
}