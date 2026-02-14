#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <set>

using namespace std;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
	vector<float> data;
	vector<float> grad;
	vector<int> shape;

	// for gradients
	vector<shared_ptr<Tensor>> parents;
	function<void()> backward_operation;

	Tensor(vector<float> d) {
		data = d;
		int size = d.size();
		shape = {size};
		grad.resize(size, 0.0f);
	}

	Tensor(vector<float> d, vector<int> s){
		data = d;
		shape = s;
		grad.resize(data.size(), 0.0f);
	}

	void backward();
};


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
shared_ptr<Tensor> add(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
	if (a->shape != b->shape) throw runtime_error("shape mismatch while adding");
	int size = a->data.size();
	
	vector<float> result_data(size);
	for (int i = 0; i < size; i++){
		result_data[i] = a->data[i] + b->data[i];
	}

	auto res = make_shared<Tensor>(move(result_data), vector<int>(a->shape));

	// for back propagation
	res->parents = {a,b};
	res->backward_operation = [res, a, b](){
		for (int i = 0; i < a->data.size(); i++){
			a->grad[i] += res->grad[i];
			b->grad[i] += res->grad[i];
		}
	};
	
	return res;
}

shared_ptr<Tensor> sub(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
	if (a->shape != b->shape) throw runtime_error("shape mismatch while subtracting");
	int size = a->data.size();
	
	vector<float> result_data(size);
	for (int i = 0; i < size; i++){
		result_data[i] = a->data[i] - b->data[i];
	}
	
	auto res = make_shared<Tensor>(move(result_data), vector<int>(a->shape));
	
	// for back propagation
	res->parents = {a,b};
	res->backward_operation = [res, a, b](){
				for (int i = 0; i < a->data.size(); i++){
					a->grad[i] += res->grad[i];
					b->grad[i] -= res->grad[i];
				}
		};

		return res;
	}

shared_ptr<Tensor> mult(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
	if (a->shape != b->shape) throw runtime_error("shape mismatch while multiplying");
	int size = a->data.size();

	vector<float> result_data(size);
	for (int i = 0; i < size; i++){
			result_data[i] = a->data[i] * b->data[i];
	}

	auto res = make_shared<Tensor>(move(result_data), vector<int>(a->shape));

	// for back propagation
	res->parents = {a,b};
	res->backward_operation = [res, a, b](){
	for (int i = 0; i < a->data.size(); i++){
			a->grad[i] += b->data[i] * res->grad[i];
			b->grad[i] += a->data[i] * res->grad[i];
		}
	};
	
	return res;
}
		
shared_ptr<Tensor> div(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
	if (a->shape != b->shape) throw runtime_error("shape mismatch while dividing");
	int size = a->data.size();
	
	vector<float> result_data(size);
	for (int i = 0; i < size; i++){
		if (b->data[i] == 0) throw runtime_error("division by zero");
		result_data[i] = a->data[i] / b->data[i];
	}

	auto res = make_shared<Tensor>(move(result_data), vector<int>(a->shape));
	
	// for back propagation
	res->parents = {a,b};
	res->backward_operation = [res, a, b](){
	for (int i = 0; i < a->data.size(); i++){
		a->grad[i] += (1.0f / b->data[i]) * res->grad[i];
			b->grad[i] -= (a->data[i] / (b->data[i] * b->data[i])) * res->grad[i];
		}
	};
	
	return res;
}
			
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	if (a->shape.size() != 2 || b->shape.size() != 2) {
		throw std::runtime_error("matmul: Both tensors must be 2D matrices.");
	}
	if (a->shape[1] != b->shape[0]) {
		throw std::runtime_error("matmul: Dimension mismatch. A.cols (" + 
			std::to_string(a->shape[1]) + ") must match B.rows (" + 
								 std::to_string(b->shape[0]) + ").");
	}

	int M = a->shape[0];
	int K = a->shape[1];
	int N = b->shape[1];
	
	std::vector<float> res_data(M * N, 0.0f);
	for (int i = 0; i < M; ++i) {
		for (int k = 0; k < K; ++k) {
			float a_val = a->data[i * K + k];
			for (int j = 0; j < N; ++j) {
				res_data[i * N + j] += a_val * b->data[k * N + j];
			}
		}
	}
	
	auto res = std::make_shared<Tensor>(std::move(res_data), std::vector<int>{M, N});
	
	res->parents = {a, b};
	
	// Derivative of C = A @ B:
	// dL/dA = (dL/dC) @ B^T
	// dL/dB = A^T @ (dL/dC)
	res->backward_operation = [res, a, b, M, K, N]() {
		for (int i = 0; i < M; ++i) {
			for (int k = 0; k < K; ++k) {
				// Pre-calculate index and value for A to minimize lookups
				int a_idx = i * K + k;
				float a_val = a->data[a_idx];
				
				for (int j = 0; j < N; ++j) {
					float grad_out = res->grad[i * N + j];
					
					// Accumulate gradient for A: sum(grad_out * B)
					a->grad[a_idx] += grad_out * b->data[k * N + j];
					
					// Accumulate gradient for B: sum(A * grad_out)
					b->grad[k * N + j] += a_val * grad_out;
				}
			}
		}
	};

	return res;
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) {
    std::vector<float> res_data(a->data.size());
    for (size_t i = 0; i < a->data.size(); ++i) {
        res_data[i] = std::max(0.0f, a->data[i]);
    }

    auto res = std::make_shared<Tensor>(std::move(res_data), a->shape);
    res->parents = {a};
    res->backward_operation = [res, a]() {
        for (size_t i = 0; i < a->data.size(); ++i) {
            if (a->data[i] > 0) {
                a->grad[i] += res->grad[i];
            }
            // else grad is 0, so we do nothing
        }
    };
    return res;
}

// Non-member operator overload
shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	return add(a, b); 
}
		
shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	return sub(a, b); 
}

shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	return mult(a, b); 
}
		
shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	return div(a, b); 
}

void Tensor::backward() {
	fill(this->grad.begin(), this->grad.end(), 1.0f);
	std::vector<std::shared_ptr<Tensor>> order;
	std::set<std::shared_ptr<Tensor>> visited;

	build_topological_order(shared_from_this(), order, visited);

	for (auto it = order.rbegin(); it != order.rend(); ++it) {
		if ((*it)->backward_operation) {
			(*it)->backward_operation();
		}
	}		
}
