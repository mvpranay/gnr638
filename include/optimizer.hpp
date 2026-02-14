#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"
#include <vector>
#include <memory>

class SGD {
public:
    std::vector<std::shared_ptr<Tensor>> params;
    float lr;

    SGD(std::vector<std::shared_ptr<Tensor>> p, float learning_rate);
    void step();
    void zero_grad();
};

#endif