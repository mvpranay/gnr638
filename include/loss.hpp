#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"
#include <memory>
#include <math.h>

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> logits, const std::vector<int>& targets);

#endif