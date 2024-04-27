#include <functional>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>


class Tensor{
    std::vector<float> data;
    std::vector<float> grad;
    float *d_data, *d_grad;
    std::tuple<Tensor*, Tensor*> _prev;
    std::string _op;
    std::string label;

    static cublasHandle_t cublasH;
    static cudaStream_t stream;
public:
    void init();

    Tensor(std::vector<float> &dataInput);

    Tensor(std::vector<float> &dataInput, std::tuple<Tensor*, Tensor*> _children, std::string _op = "", std::string label = "");

    Tensor(std::tuple<Tensor*, Tensor*> _children, std::string _op = "", std::string label = "");

    Tensor(int dataSize, std::tuple<Tensor*, Tensor*>_children, std::string _op);

    ~Tensor();

    void  _backward(float* d_parentGrad = nullptr, int parentGradLen = 0);

    std::string repr();

    Tensor& operator+(Tensor &other);

    Tensor& operator-(Tensor &other);

    Tensor& operator-();

    Tensor& dot(Tensor &other);

    void assignGradient(const std::vector<float> &gradInput);
    
    std::vector<float>& getGrad();

    void detach();

};