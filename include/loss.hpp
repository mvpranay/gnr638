#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"
#include <memory>

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);

#endif