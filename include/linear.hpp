#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "tensor.hpp"
#include "ops.hpp"
#include <vector>
#include <memory>

class Linear {
public:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;

    Linear(int in_features, int out_features);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
    std::vector<std::shared_ptr<Tensor>> parameters();

    int param_count() {
        return weights->data.size() + bias->data.size();
    }
};

#endif
