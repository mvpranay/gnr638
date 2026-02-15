#include "tensor.hpp"
#include "ops.hpp"

Tensor::Tensor(std::vector<float> d) {
    data = d;
    int size = d.size();
    shape = {size};
    grad.resize(size, 0.0f);
}

Tensor::Tensor(std::vector<float> d, std::vector<int> s){
    data = d;
    shape = s;
    grad.resize(data.size(), 0.0f);
}

void build_topological_order(std::shared_ptr<Tensor> v, 
							  std::vector<std::shared_ptr<Tensor>>& order, 
							  std::set<std::shared_ptr<Tensor>>& visited) {
	if (v == nullptr) return;
	
	visited.insert(v);
	for (auto& parent : v->parents) {
		if (visited.find(parent) == visited.end()) 
		build_topological_order(parent, order, visited);
	}
	order.push_back(v);
}

void Tensor::backward() {
	std::fill(this->grad.begin(), this->grad.end(), 1.0f);
	std::vector<std::shared_ptr<Tensor>> order;
	std::set<std::shared_ptr<Tensor>> visited;

	build_topological_order(shared_from_this(), order, visited);

	for (auto it = order.rbegin(); it != order.rend(); ++it) {
		if ((*it)->backward_operation) {
			(*it)->backward_operation();
		}
	}		
}

std::shared_ptr<Tensor> Tensor::flatten() {
    // flattens everything except the batch dimension (index 0)
    int batch_size = this->shape[0];
    int flat_features = (int)this->data.size() / batch_size;
    return this->view({batch_size, flat_features});
}

std::shared_ptr<Tensor> Tensor::view(std::vector<int> new_shape) {
    // verify number of elements remaining the same
    int new_size = 1;
    for (int s : new_shape) new_size *= s;
    if (new_size != (int)this->data.size()) {
        throw std::runtime_error("view : size mismatch");
    }

    auto res = std::make_shared<Tensor>(this->data, new_shape);
    
    res->parents = { shared_from_this() };
    res->backward_operation = [res, self = shared_from_this()]() {
        for (size_t i = 0; i < self->grad.size(); i++) {
            self->grad[i] += res->grad[i];
        }
    };
    return res;
}

// Non-member operator overload
std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	return add(a, b); 
}
		
std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	return sub(a, b); 
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	return mult(a, b); 
}
		
std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	return div(a, b); 
}

std::shared_ptr<Tensor> Tensor::operator+=(std::shared_ptr<Tensor> other) {
    return add(std::shared_ptr<Tensor>(this), other); 
}

std::shared_ptr<Tensor> Tensor::operator-=(std::shared_ptr<Tensor> other) {
    return sub(std::shared_ptr<Tensor>(this), other); 
}

std::shared_ptr<Tensor> Tensor::operator*=(std::shared_ptr<Tensor> other) {
    return mult(std::shared_ptr<Tensor>(this), other); 
}

std::shared_ptr<Tensor> Tensor::operator/=(std::shared_ptr<Tensor> other) {
    return div(std::shared_ptr<Tensor>(this), other); 
}
