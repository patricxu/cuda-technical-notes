#include "testcase.hpp"
#include "tensor.hpp"


void checkGrad(string name, Tensor &tensor) {
    std::cout << name <<  "Gradient: " << endl;
    for(auto element : tensor.getGrad()) {
        std::cout << element << ", ";
    }
    std::cout << std::endl;
}


void testCase1() {
    cout << "testcase1 vector add" << endl;
    std::vector<float> data1 = {1, 2, 3, 4};
    std::vector<float> data2 = {5, 6, 7, 8};
    Tensor x(data1);
    Tensor y(data2);
    Tensor z = x + y;
    z.assignGradient({1, 1, 1, 1});
    z._backward();
    z._backward();
    x.detach();
    checkGrad("x", x);
    y.detach();
    checkGrad("y", y);
}


void testCase2() {
    cout << "testcase2 vector dot" << endl;
    std::vector<float> data1 = {1, 2, 3, 4};
    std::vector<float> data2 = {5, 6, 7, 8};
    Tensor x(data1);
    Tensor y(data2);

    Tensor xdoty = x.dot(y);
    xdoty.assignGradient({1});
    xdoty._backward();
    xdoty.detach();
    std::cout << "Dot product: " << xdoty.repr() << std::endl;

    x.detach();
    checkGrad("x", x);
    y.detach();
    checkGrad("y", y);

    std::cout << "scond backward" << endl;
    xdoty._backward();
    x.detach();
    checkGrad("x", x);
    y.detach();
    checkGrad("y", y);
}


void testCase3() {
    cout << "testcase3 vector dot with parent gradient" << endl;
    std::vector<float> data1 = {1, 2, 3, 4};
    std::vector<float> data2 = {5, 6, 7, 8};
    Tensor x(data1);
    Tensor y(data2);
    Tensor z = x.dot(y);
    float parentGrad[] = {2, 2, 2, 2};
    z.assignGradient({1});
    z._backward(parentGrad, 4);
    x.detach();
    checkGrad("x", x);
    y.detach();
    checkGrad("y", y);
}

