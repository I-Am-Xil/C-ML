/*
Hay tantos pesos en la transformacion polynomial como inputs
Queda atrapado en muchos minimos locales
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct{

    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;

} xor;

typedef struct{

    float wp_or1;
    float wp_or2;

    float wp_nand1;
    float wp_nand2;

    float wp_and1;
    float wp_and2;

} pol;

float sigf(float x){
    return 1.f/(1.f + expf(-x));
}

float rand_float(void){
    return (float) rand()/ (float) RAND_MAX;
}

typedef float sample[3];

sample xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

sample *train = xor_train;
size_t train_count = 4;


float polyTrans(float y, float wp1, float wp2){
    return y*(wp1 + wp2*y);
}

float forward(xor m, float x1, float x2, pol p){
    float a = sigf(polyTrans(m.or_w1*x1 + m.or_w2*x2 + m.or_b, p.wp_or1, p.wp_or2));
    float b = sigf(polyTrans(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b, p.wp_nand1, p.wp_nand2));
    return sigf(polyTrans(a*m.and_w1 + b*m.and_w2 + m.and_b, p.wp_and1, p.wp_and2));
}

float cost(xor m, pol p){
    float result = 0.0f;

    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2, p);
        float d = y - train[i][2];

        result += d*d;
    }

    result /= train_count;
    return result;
}

xor rand_xor(void){
    xor m;

    m.or_w1 = rand_float(); 
    m.or_w2 = rand_float(); 
    m.or_b = rand_float(); 

    m.nand_w1 = rand_float(); 
    m.nand_w2 = rand_float(); 
    m.nand_b = rand_float(); 

    m.and_w1 = rand_float(); 
    m.and_w2 = rand_float(); 
    m.and_b = rand_float(); 

    return m;
}

pol rand_p(void){
    pol p;

    p.wp_or1 = rand_float();
    p.wp_or2 = rand_float();

    p.wp_nand1 = rand_float();
    p.wp_nand2 = rand_float();

    p.wp_and1 = rand_float();
    p.wp_and2 = rand_float();

    return p;
}

void print_xor(xor m){
    printf("or_w1 = %f\n", m.or_w1);
    printf("or_w2 = %f\n", m.or_w2);
    printf("or_b = %f\n", m.or_b);
    printf("nand_w1 = %f\n", m.nand_w1);
    printf("nand_w2 = %f\n", m.nand_w2);
    printf("nand_b = %f\n", m.nand_b);
    printf("and_w1 = %f\n", m.and_w1);
    printf("and_w2 = %f\n", m.and_w2);
    printf("and_b = %f\n", m.and_b);
}

xor learn1(xor m, xor g, float rate){

    m.or_w1 -= rate*g.or_w1;
    m.or_w2 -= rate*g.or_w2;
    m.or_b -= rate*g.or_b;
    m.nand_w1 -= rate*g.nand_w1;
    m.nand_w2 -= rate*g.nand_w2;
    m.nand_b -= rate*g.nand_b;
    m.and_w1 -= rate*g.and_w1;
    m.and_w2 -= rate*g.and_w2;
    m.and_b -= rate*g.and_b;

    return m;

}

pol learn2(pol p, pol q, float rate){

    p.wp_or1 -= rate*q.wp_or1;
    p.wp_or2 -= rate*q.wp_or2;
    p.wp_nand1 -= rate*q.wp_nand1;
    p.wp_nand2 -= rate*q.wp_nand2;
    p.wp_and1 -= rate*q.wp_and1;
    p.wp_and2 -= rate*q.wp_and2;

    return p;
}

xor finite_diff(xor m, pol p, float eps){

    xor g;
    float c = cost(m, p);
    float saved;

    saved = m.or_w1; 
    m.or_w1 += eps;
    g.or_w1 = (cost(m, p) - c)/eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m, p) - c)/eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m, p) - c)/eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m, p) - c)/eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m, p) - c)/eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m, p) - c)/eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m, p) - c)/eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m, p) - c)/eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m, p) - c)/eps;
    m.and_b = saved;

    return g;
}

pol finite_diffp(xor m, pol p, float eps){

    pol g;
    float c = cost(m, p);
    float saved;

    saved = p.wp_or1;
    p.wp_or1 += eps;
    g.wp_or1 = (cost(m, p) - c)/eps;
    p.wp_or1 = saved;

    saved = p.wp_or2;
    p.wp_or2 += eps;
    g.wp_or2 = (cost(m, p) - c)/eps;
    p.wp_or2 = saved;

    saved = p.wp_nand1;
    p.wp_nand1 += eps;
    g.wp_nand1 = (cost(m, p) - c)/eps;
    p.wp_nand1 = saved;

    saved = p.wp_nand2;
    p.wp_nand2 += eps;
    g.wp_nand2 = (cost(m, p) - c)/eps;
    p.wp_nand2 = saved;

    saved = p.wp_and1;
    p.wp_and1 += eps;
    g.wp_and1 = (cost(m, p) - c)/eps;
    p.wp_and1 = saved;

    saved = p.wp_and2;
    p.wp_and2 += eps;
    g.wp_and2 = (cost(m, p) - c)/eps;
    p.wp_and2 = saved;

    return g;
}

int main(){
    srand(time(0));
    float rate = 3e-1;
    float eps = 1e-1;

    xor m = rand_xor();
    pol p = rand_p();


    printf("%f\n", cost(m, p));
    for (size_t i = 0; i < 10000; i++) {
        xor g = finite_diff(m, p, eps);
        pol q = finite_diffp(m, p, eps);
        m = learn1(m, g, rate);
        p = learn2(p, q, rate);
    }
    printf("%f\n", cost(m, p));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%zu ^ %zu = %f\n", i, j, forward(m, i, j, p));
        }
    }

    return 0;
}
