#include "linear.hpp"
#include "init.hpp"

Linear::Linear(int in_features, int out_features) {
    weights = xavier_init({in_features, out_features});
    bias = std::make_shared<Tensor>(std::vector<float>(out_features, 0.0f), 
                                     std::vector<int>{1, out_features});
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    return matmul(input, weights) + bias;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    return {weights, bias};
}

int Linear::param_count() {
    return weights->data.size() + bias->data.size();
}