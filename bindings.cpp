#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "tensor.hpp"
#include "linear.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include "ops.hpp"
#include "init.hpp"
#include "conv.hpp"

namespace py = pybind11;

PYBIND11_MODULE(APDNN, m) {
    m.doc() = "C++ CNN Backend";

    // Tensor class - keep_alive ensures Python holds reference
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<float>, std::vector<int>>())
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("shape", &Tensor::shape)
        .def("backward", &Tensor::backward)
        .def("zero_grad", [](std::shared_ptr<Tensor> self) {
            std::fill(self->grad.begin(), self->grad.end(), 0.0f);
        })
        .def("__repr__", [](const Tensor &t) {
            std::string s = "<Tensor shape=[";
            for (size_t i = 0; i < t.shape.size(); i++) {
                s += std::to_string(t.shape[i]);
                if (i < t.shape.size() - 1) s += ", ";
            }
            s += "]>";
            return s;
        })
        // Overloading operators for Python
        .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return add(a, b); })
        .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return sub(a, b); })
        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return mult(a, b); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return div(a, b); })
        .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return matmul(a, b); })
        .def("view", &Tensor::view)
        .def("flatten", &Tensor::view);

    // Tensor operators - use lambdas to avoid ambiguity
    m.def("add", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
        return add(a, b); 
    }, "Element-wise addition");
    
    m.def("sub", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
        return sub(a, b); 
    }, "Element-wise subtraction");
    
    m.def("mult", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
        return mult(a, b); 
    }, "Element-wise multiplication");
    
    m.def("div", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
        return div(a, b); 
    }, "Element-wise division");
    
    m.def("matmul", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
        return matmul(a, b); 
    }, "Matrix multiplication");
    
    m.def("relu", [](std::shared_ptr<Tensor> a) { 
        return relu(a); 
    }, "ReLU activation");

    // Linear layer - return_value_policy ensures proper reference
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int>())
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters)
        .def_readwrite("weights", &Linear::weights)  
        .def_readwrite("bias", &Linear::bias);        

    // Conv2D layer
    py::class_<Conv2D>(m, "Conv2D")
        .def(py::init<int, int, int, int, int>(), 
             py::arg("in_channels"), 
             py::arg("out_channels"), 
             py::arg("kernel_size"), 
             py::arg("stride") = 1, 
             py::arg("padding") = 0)
        .def("forward", &Conv2D::forward)
        .def("parameters", &Conv2D::parameters)
        .def_readwrite("weights", &Conv2D::weights)
        .def_readwrite("bias", &Conv2D::bias)
        .def_readwrite("stride", &Conv2D::stride)
        .def_readwrite("padding", &Conv2D::padding);

    // Optimizer
    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<std::shared_ptr<Tensor>>, float>())
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad)
        .def_readwrite("lr", &SGD::lr);

    // Loss functions
    m.def("mse_loss", [](std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) { 
        return mse_loss(pred, target); 
    }, "Mean Squared Error loss");
    
    m.def("set_random_seed", &set_random_seed, "Set random seed for initialization");
}