#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include "tensor.h" // Ensure your Tensor class and functions are in this header

void print_tensor(const std::string& name, std::shared_ptr<Tensor> t) {
    std::cout << name << " (data): ";
    for (float val : t->data) std::cout << val << " ";
    std::cout << "\n" << name << " (grad): ";
    for (float val : t->grad) std::cout << val << " ";
    std::cout << "\n\n";
}

int main() {
    // 1. Test Basic Element-wise Ops: (a * b) + (c - d) / e
    auto a = std::make_shared<Tensor>(std::vector<float>{2.0, 3.0});
    auto b = std::make_shared<Tensor>(std::vector<float>{4.0, 5.0});
    auto c = std::make_shared<Tensor>(std::vector<float>{10.0, 12.0});
    auto d = std::make_shared<Tensor>(std::vector<float>{2.0, 2.0});
    auto e = std::make_shared<Tensor>(std::vector<float>{2.0, 2.0});

    // C++ Overloaded operations
    auto mul_res = a * b;             // {8, 15}
    auto sub_res = c - d;             // {8, 10}
    auto div_res = sub_res / e;       // {4, 5}
    auto final_add = mul_res + div_res; // {12, 20}

    std::cout << "--- Element-wise Test ---" << std::endl;
    final_add->backward();
    
    print_tensor("A", a); // Grad should be B * 1 = {4, 5}
    print_tensor("E", e); // Grad should be -(c-d)/e^2 = -8/4, -10/4 = {-2, -2.5}
    print_tensor("Final", final_add);

    // 2. Test MatMul: X @ W
    // X (1 x 3) @ W (3 x 2) = Output (1 x 2)
    auto x = std::make_shared<Tensor>(
        std::vector<float>{1.0, 2.0, 3.0}, 
        std::vector<int>{1, 3}
    );
    auto w = std::make_shared<Tensor>(
        std::vector<float>{1.0, 0.0, 
                           0.0, 1.0, 
                           1.0, 1.0}, 
        std::vector<int>{3, 2}
    );

    auto out = matmul(x, w); // Result: { (1*1 + 2*0 + 3*1), (1*0 + 2*1 + 3*1) } = {4, 5}

    std::cout << "--- MatMul Test ---" << std::endl;
    out->backward();

    print_tensor("X", x); 
    // Grad wrt X = out_grad @ W^T = [1, 1] @ [[1, 0, 1], [0, 1, 1]] = [1, 1, 2]
    
    print_tensor("W", w);
    // Grad wrt W = X^T @ out_grad
    
    print_tensor("Out", out);

    std::cout << "All tests completed!" << std::endl;

    return 0;
}