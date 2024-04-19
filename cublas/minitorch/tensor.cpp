#include <functional>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "kernels.hpp"
#include "tensor.hpp"


cublasHandle_t Tensor::cublasH = nullptr;
cudaStream_t Tensor::stream = nullptr;

void Tensor::init() {
    if (cublasH == nullptr) {
        // Create a cuBLAS handle
        CUBLAS_CHECK(cublasCreate(&cublasH));
    }
    if (stream == nullptr) {
        // Create a stream
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    
    grad.assign(data.size(), 1);
    
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_data, data.size() * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_grad, grad.size() * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_grad, 0, grad.size() * sizeof(float), stream));
}

Tensor::Tensor(std::vector<float> &dataInput): data(dataInput), grad(), _prev(), _op(""), label("") {
        init();
    }

Tensor::Tensor(std::vector<float> &dataInput, std::tuple<Tensor*, Tensor*> _children, std::string _op, std::string label)
    : data(dataInput), grad(), _prev(_children), _op(_op), label(label) {
        cout << "Data input: " << dataInput.size() << endl;
        init();
    }

Tensor::Tensor(std::tuple<Tensor*, Tensor*> _children, std::string _op, std::string label)
    : data(), grad(), _prev(_children), _op(_op), label(label) {
        init();
    }

Tensor::Tensor(int dataSize, std::tuple<Tensor*, Tensor*>_children, std::string _op): _prev(_children), _op(_op), label("") {
        data.assign(dataSize, 0);
        init();
    }

Tensor::~Tensor() {
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_grad));
}

void Tensor::_backward(std::vector<float> gradInput) {
    if (gradInput.size() > 0) {
        assignGradient(gradInput);
    }

    auto left = std::get<0>(_prev);
    auto right = std::get<1>(_prev);

    if (_op == "+") {
        float alpha = 1.0;
        CUBLAS_CHECK(cublasSaxpy(cublasH, grad.size(), &alpha, d_grad, 1, left->d_grad, 1));
        CUBLAS_CHECK(cublasSaxpy(cublasH, grad.size(), &alpha, d_grad, 1, right->d_grad, 1));
    }
    else if (_op == "dot") {
        const float *scalar = d_grad;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(left->d_grad, right->d_data, right->data.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        scalarMulVect(scalar, left->d_grad, left->grad.size(), stream);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(right->d_grad, left->d_data, left->data.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        scalarMulVect(scalar, right->d_grad, right->grad.size(), stream);
    }
    else if (_op == "-")
    {
        if (right != nullptr) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(left->d_grad, d_grad, grad.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(right->d_grad, d_grad, grad.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        else {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(left->d_grad, d_grad, grad.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
    }
    
}


std::string Tensor::repr() {
    std::string dataStr;
    for (const auto& element : this->data) {
        dataStr += std::to_string(element) + ", ";
    }
    return "Tensor(data=" + dataStr + ")";
}

Tensor& Tensor::operator+(Tensor &other) {
    float alpha = 1.0;
    Tensor *out = new Tensor(other.data, std::tuple{this, &other}, "+");
    CUBLAS_CHECK(cublasSaxpy_v2(cublasH, this->data.size(), &alpha, d_data, 1, out->d_data, 1));
    return *out;
}

Tensor& Tensor::operator-(Tensor &other) {
    float alpha = -1.0;
    Tensor *out = new Tensor(other.data, std::tuple{this, &other}, "-");
    CUBLAS_CHECK(cublasSaxpy_v2(cublasH, this->data.size(), &alpha, d_data, 1, out->d_data, 1));
    return *out;
}

Tensor& Tensor::operator-() {
    float scalar = -1.0;
    scalarMulVect(&scalar, d_data, data.size(), stream);
    return *this;
}

Tensor& Tensor::dot(Tensor &other) {
    float alpha = 1.0;
    Tensor *out = new Tensor(1, std::tuple{this, &other}, "dot");
    CUBLAS_CHECK(cublasSdot(cublasH, this->data.size(), d_data, 1, other.d_data, 1, out->d_data));
    return *out;
}

void Tensor::assignGradient(const std::vector<float> &gradInput) {
    grad = gradInput;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_grad, grad.data(), grad.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
}

vector<float>& Tensor::getGrad() {
    return grad;
}

void Tensor::detach() {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(data.data(), d_data, data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(grad.data(), d_grad, data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}



