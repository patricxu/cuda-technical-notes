#include "testcase.hpp"
#include "tensor.hpp"


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
    cout << "x gradient" << endl;
    for(auto element : x.getGrad()) {
        std::cout << element << ", ";
    }
    std::cout << std::endl;
    y.detach();
    cout << "y gradient" << endl;
    for(auto element : y.getGrad()) {
        std::cout << element << ", " << std::endl;
    }
    std::cout << std::endl;
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
    cout << "x gradient" << endl;
    for(auto element : x.getGrad()) {
        std::cout << element << ", ";
    }
    std::cout << std::endl;
    y.detach();
    cout << "y gradient" << endl;
    for(auto element : y.getGrad()) {
        std::cout << element << ", ";
    }
    std::cout << std::endl;
}