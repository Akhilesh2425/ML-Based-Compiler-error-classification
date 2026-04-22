#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int result;
    result = add(5);

    printf("Result = %d\n", result);
    return 0;
}