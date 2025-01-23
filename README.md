# Particle Swarm Optimizer (PSO) Feed Forward Neural Network for Sine Approximation

## Project Overview

This project implements a Particle Swarm Optimizer (PSO) to train a Feed Forward Neural Network (FFNN) for the task of approximating the sine function. The main objective is to explore the capabilities of PSO in optimizing the weights of a neural network and achieving an accurate approximation of the sine wave. The problem is tackled by training a neural network to predict values of the sine function given a set of input values using a Mean Squared Error for particle fitness.

### Key Components:
1. **Particle Swarm Optimization (PSO)**: A population-based optimization algorithm inspired by the social behavior of birds flocking or fish schooling. PSO is used to optimize the weights of the neural network.
2. **Feed Forward Neural Network (FFNN)**: An artificial neural network where the connections between the nodes do not form cycles. The FFNN is used to approximate the sine function.
3. **Sine Approximation**: The goal is to approximate the mathematical sine function using the trained neural network.

## Problem Definition

Given an input `x`, the task is to predict the sine of `x` (i.e., `sin(x)`) using a neural network. The optimization process is performed using PSO, where the particles represent different potential sets of weights for the network, and the fitness function evaluates the network's performance.

## Approach

### 1. **Feed Forward Neural Network (FFNN)**
- A simple FFNN with one hidden layer is used.
- The input layer consists of a single neuron (for the scalar input `x`).
- The output layer has a single neuron representing the sine approximation.
- A non-linear activation function (e.g., sigmoid) is applied in the hidden layer to introduce non-linearity.

### 2. **Particle Swarm Optimization (PSO)**
- PSO is used to optimize the weights and biases of the neural network.
- Each particle in the swarm represents a possible set of weights.
- The fitness of each particle is determined by how accurately the neural network approximates the sine function.
- PSO parameters such as inertia weight, cognitive and social coefficients, and swarm size are fine-tuned for optimal performance.

### 3. **Training & Evaluation**
- The neural network is trained on a dataset of input-output pairs of `x` and `sin(x)` values.
- PSO iteratively updates the particle positions (weights) to minimize the error between the neural network's predictions and the sine values.
- The performance is evaluated using Mean Squared Error (MSE) between predicted and actual sine values.

## Results

- The neural network, once trained, should demonstrate a high level of accuracy in predicting the sine function over a range of input values.
- The optimization process shows how PSO can effectively tune the network's weights to approximate complex functions like sine.

## Visualizing the Change in Global Best Particle Using Matplotlib-cpp

In this project, we can track and visualize the progress of the PSO algorithm by plotting how the **global best** particle evolves. The global best particle represents the best solution found so far across all particles in the swarm. This visualization helps us understand the convergence behavior of the optimization process.

I used the **matplotplusplus** library to generate visualizations similar to Python's `matplotlib`. By plotting the predicted sin wave of the global best particle against the actual sin wave I can visualize how well the global best particle is forming to the expected output. Along with that the velocity of the particle is ploted in black to represent how much movement is occuring between iterations.

## Global Best Visualization

<div align="center">
    <img src="https://isoptera.lcsc.edu/~rjziegler/sine-approx.png" width=800px/>
</div>

## Final Weights

<div align="center">
    <img src="https://isoptera.lcsc.edu/~rjziegler/neural-network.png" width=800px/>
</div>
