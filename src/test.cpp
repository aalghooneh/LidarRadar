#include <iostream>
#include <omp.h>

int main() {
    const int n = 100;
    int sum = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        sum += i;
    }

    std::cout << "Sum is: " << sum << std::endl;

    return 0;
}