#ifndef INIT_HPP
#define INIT_HPP

#include "tensor.hpp"
#include <vector>
#include <memory>

std::shared_ptr<Tensor> xavier_init(std::vector<int> shape);

#endif