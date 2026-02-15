#include "conv.hpp"
#include "init.hpp"

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int s, int p) : stride(s), padding(p) {
    std::vector<int> kernel_shape = {out_channels, in_channels, kernel_size, kernel_size};
    // initialize weights
    weights = xavier_init(kernel_shape);

    // initialize biases
    std::vector<float> b_data(out_channels, 0.0f);
    bias = std::make_shared<Tensor>(std::move(b_data), std::vector<int>({out_channels}));
}

std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {
    return conv2d(input, weights, bias, stride, padding);
}

std::vector<std::shared_ptr<Tensor>> Conv2D::parameters() {
    return {weights, bias};
}