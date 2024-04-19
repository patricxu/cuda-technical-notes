#include <functional>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "kernels.hpp"


// Create a cuBLAS handle
cublasHandle_t cublasH;
cudaStream_t stream;

class Tensor{
    std::vector<float> data;
    std::vector<float> grad;
    float *d_data, *d_grad;
    std::tuple<Tensor*, Tensor*> _prev;
    std::string _op;
    std::string label;
public:
    void (*backwardfunc)(void);

    void init() {
        grad.assign(data.size(), 1);
        
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_data, data.size() * sizeof(float), stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_grad, grad.size() * sizeof(float), stream));
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_grad, 0, grad.size() * sizeof(float), stream));
    }

    Tensor(std::vector<float> &dataInput): data(dataInput), grad(), _prev(), _op(""), label("") {
            init();
        }

    Tensor(std::vector<float> &dataInput, std::tuple<Tensor*, Tensor*> _children, std::string _op = "", std::string label = "")
        : data(dataInput), grad(), _prev(_children), _op(_op), label(label) {
            cout << "Data input: " << dataInput.size() << endl;
            init();
        }

    Tensor(std::tuple<Tensor*, Tensor*> _children, std::string _op = "", std::string label = "")
        : data(), grad(), _prev(_children), _op(_op), label(label) {
            init();
        }

    Tensor(int dataSize, std::tuple<Tensor*, Tensor*>_children, std::string _op): _prev(_children), _op(_op), label("") {
            data.assign(dataSize, 0);
            init();
        }

    ~Tensor() {
        CHECK_CUDA_ERROR(cudaFree(d_data));
        CHECK_CUDA_ERROR(cudaFree(d_grad));
    }

    void  _backward(std::vector<float> gradInput = {}) {
        if (gradInput.size() > 0) {
            assignGradient(gradInput);
        }
        
        if (_op == "+") {
            auto left = std::get<0>(_prev);
            auto right = std::get<1>(_prev);

            CHECK_CUDA_ERROR(cudaMemcpyAsync(left->d_grad, d_grad, grad.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(right->d_grad, d_grad, grad.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        else if (_op == "dot") {
            auto left = std::get<0>(_prev);
            auto right = std::get<1>(_prev);
            const float *scalar = d_grad;
            CHECK_CUDA_ERROR(cudaMemcpyAsync(left->d_grad, right->d_data, right->data.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            // CUBLAS_CHECK(cublasSaxpy_v2(cublasH, left->grad.size(), &alpha, d_grad, 1, left->d_grad, 1));
            scalarMulVect(scalar, left->d_grad, left->grad.size(), stream);

            CHECK_CUDA_ERROR(cudaMemcpyAsync(right->d_grad, left->d_data, left->data.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            // CUBLAS_CHECK(cublasSaxpy_v2(cublasH, right->grad.size(), &alpha, d_grad, 1, right->d_grad, 1));
            scalarMulVect(scalar, right->d_grad, right->grad.size(), stream);
        }
    }


    std::string repr() {
        std::string dataStr;
        for (const auto& element : this->data) {
            dataStr += std::to_string(element) + ", ";
        }
        return "Tensor(data=" + dataStr + ")";
    }

    Tensor& operator+(Tensor &other) {
        float alpha = 1.0;
        Tensor *out = new Tensor(other.data, std::tuple{this, &other}, "+");
        CUBLAS_CHECK(cublasSaxpy_v2(cublasH, this->data.size(), &alpha, d_data, 1, out->d_data, 1));
        return *out;
    }

    Tensor& dot(Tensor &other) {
        float alpha = 1.0;
       Tensor *out = new Tensor(1, std::tuple{this, &other}, "dot");
        CUBLAS_CHECK(cublasSdot(cublasH, this->data.size(), d_data, 1, other.d_data, 1, out->d_data));
        return *out;
    }

    void assignGradient(const std::vector<float> &gradInput) {
        grad = gradInput;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_grad, grad.data(), grad.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    vector<float>& getGrad() {
        return grad;
    }

    void detach() {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(data.data(), d_data, data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(grad.data(), d_grad, data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }



    // Tensor operator/(Tensor other) {
    //     return *this * other.pow(-1);
    // }

    // Tensor operator-() {
    //     return *this * -1;
    // }

    // Tensor operator-(Tensor other) {
    //     return *this + (-other);
    // }

    // Tensor pow(double power) {
    //     Tensor out = Tensor(this->data.array().pow(power), {this}, "pow");
    //     out._backward = [this, &out, power]() {
    //         this->grad += power * this->data.array().pow(power - 1) * out.grad;
    //     };
    //     return out;
    // }

    // void backward() {
    //     std::vector<Tensor*> topo;
    //     std::set<Tensor*> visited;
    //     std::function<void(Tensor*)> build_topo = [&visited, &topo, &build_topo](Tensor* v) {
    //         if (visited.find(v) == visited.end()) {
    //             visited.insert(v);
    //             for (Tensor* child : v->_prev) {
    //                 build_topo(child);
    //             }
    //             topo.push_back(v);
    //         }
    //     };
    //     build_topo(this);
    //     this->grad = Eigen::VectorXd::Ones(this->data.size());
    //     for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    //         (*it)->_backward();
    //     }
    // }
};


class MyClass {
public:
    std::vector<float> myVector;

    MyClass(std::vector<float> inputVector) : myVector(inputVector) {
        // constructor body
    }
};


int main(int argc, char **argv) {
    // Create a cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // Create a stream
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<float> data1 = {1, 2, 3, 4};
    std::vector<float> data2 = {5, 6, 7, 8};
    Tensor x(data1);
    Tensor y(data2);
    Tensor z = x + y;
    // z.assignGradient({1, 1, 1, 1});
    // z._backward();
    // x.detach();
    // for(auto element : x.getGrad()) {
    //     std::cout << element << ", " << std::endl;
    // }

    Tensor xdoty = x.dot(y);
    xdoty.assignGradient({1});
    xdoty._backward();
    xdoty.detach();
    std::cout << "Dot product: " << xdoty.repr() << std::endl;
    x.detach();
    for(auto element : x.getGrad()) {
        std::cout << "x gradient " << element << ", " << std::endl;
    }
    std::cout << std::endl;
    y.detach();
    for(auto element : y.getGrad()) {
        std::cout << "y gradient " << element << ", " << std::endl;
    }
    std::cout << std::endl;
    return 0;
}