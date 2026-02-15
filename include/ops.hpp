#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> mult(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> kernel, std::shared_ptr<Tensor> bias, int stride, int padding);
std::shared_ptr<Tensor> maxpool2d(std::shared_ptr<Tensor> a, int pool_size = 2, int stride = 2);

#endif