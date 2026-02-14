#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <set>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
	std::vector<float> data;
	std::vector<float> grad;
	std::vector<int> shape;

	// for gradients
	std::vector<std::shared_ptr<Tensor>> parents;
	std::function<void()> backward_operation;

	Tensor(std::vector<float> d);
	Tensor(std::vector<float> d, std::vector<int> s);
	void backward();

	std::shared_ptr<Tensor> operator+=(std::shared_ptr<Tensor> other);
	std::shared_ptr<Tensor> operator-=(std::shared_ptr<Tensor> other);
	std::shared_ptr<Tensor> operator*=(std::shared_ptr<Tensor> other);
	std::shared_ptr<Tensor> operator/=(std::shared_ptr<Tensor> other);
};


// Non-member operator overload
std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);


#endif