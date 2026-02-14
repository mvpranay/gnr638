#include "tensor.h"
#include <random>

shared_ptr<Tensor> xavier_init(vector<int> shape) {
    int fan_in = shape[0];
    int fan_out = (shape.size() > 1) ? shape[1] : 1;
    float range = sqrt(6.0f / (fan_in + fan_out));

    static mt19937 gen(42); // Fixed seed for reproducibility
    uniform_real_distribution<float> dist(-range, range);

    int total_size = 1;
    for (int s : shape) total_size *= s;

    vector<float> data(total_size);
    for (int i = 0; i < total_size; i++) data[i] = dist(gen);

    return make_shared<Tensor>(move(data), shape);
}

class Linear {
public:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;

    Linear(int in_features, int out_features) {
        weights = xavier_init({in_features, out_features});
        // initializing biases to 0
        bias = std::make_shared<Tensor>(std::vector<float>(out_features, 0.0f), std::vector<int>{1, out_features});
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        // output = (input @ weights) + bias
        return matmul(input, weights) + bias;
    }

    // This returns the parameters for the optimizer to update
    std::vector<std::shared_ptr<Tensor>> parameters() {
        return {weights, bias};
    }
};