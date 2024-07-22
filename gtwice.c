#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

// y = x*w+b;
#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float(void){
    return (float) rand()/ (float) RAND_MAX;
}

float cost(float w, float b){
    float result = 0.0f;

    for (size_t i = 0; i < train_count; i++) {
        float x = train[i][0];
        float y = x*w + b;
        float d = y - train[i][1];

        result += d*d;
    }

    result /= train_count;
    return result;
}


float wcost(float w, float b){
    float result = 0.f;
    size_t n = train_count;
    for(size_t i = 0; i < n; i++){
        float y = train[i][1];
        float x = train[i][0];
        result += 2*(x*w + b - y)*x;
    }
    return result /= n;
}

float dcost(float w, float b){
    float result = 0.f;
    size_t n = train_count;
    for(size_t i = 0; i < n; i++){
        float y = train[i][1];
        float x = train[i][0];
        result += 2*(x*w + b - y);
    }
    return result /= n;
}

int main() {
    srand(time(0));
    float w = rand_float()*10.0f;
    float b = rand_float()*5.0f;

    float rate = 1e-3;

    for (size_t steps = 0; steps < 5000; steps++) {

        float dw = wcost(w, b);
        float db = dcost(w, b);

        w -= rate * dw;
        b -= rate * db;
        printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
    }

    printf("-----------------------\n");
    printf("w = %f, b = %f\n", w, b);

    return 0;
}
