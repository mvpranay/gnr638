#ifndef CONV_HPP
#define CONV_HPP

#include "tensor.hpp"
#include "ops.hpp"
#include <vector>
#include <memory>

class Conv2D {
public:
    std::shared_ptr<Tensor> weights; // shape (out_channels, in_channels, k_h, k_w)
    std::shared_ptr<Tensor> bias; // shape (out_channels,)

    int stride, padding;

    Conv2D(int in_channels, int out_channels, int kernel_size, int s = 1, int p = 0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

    std::vector<std::shared_ptr<Tensor>> parameters();
};

#endif