#include <math.h>
#define NN_IMPLEMENTATION
#include "lib/nn.h"
#include <time.h>



typedef struct {
    Mat a0, a1, a2;
    Mat w1, b1;
    Mat wp1, ap1;
    Mat w2, b2;
    Mat wp2, ap2;
} Xor;

Xor xor_alloc(void){

    Xor m;
    m.a0 = mat_alloc(1, 2);

    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);

    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);

    return m;
}

void forward_xor(Xor m){

    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);

}

float cost(Xor m, Mat ti, Mat to){
    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);
    size_t n = ti.rows;

    float c = 0;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m.a0, x);
        forward_xor(m);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++) {
            float d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;

}

void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to){
    float saved;

    float c = cost(m, ti, to);

    for(size_t i = 0; i < m.w1.rows; i++){
        for(size_t j = 0; j < m.w1.cols; j++){
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for(size_t i = 0; i < m.b1.rows; i++){
        for(size_t j = 0; j < m.b1.cols; j++){
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.b1, i, j) = saved;
        }
    }

    for(size_t i = 0; i < m.w2.rows; i++){
        for(size_t j = 0; j < m.w2.cols; j++){
            saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.w2, i, j) = saved;
        }
    }

    for(size_t i = 0; i < m.b2.rows; i++){
        for(size_t j = 0; j < m.b2.cols; j++){
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.b2, i, j) = saved;
        }
    }
}

void xor_learn(Xor m, Xor g, float rate){

    for(size_t i = 0; i < m.w1.rows; i++){
        for(size_t j = 0; j < m.w1.cols; j++){
            MAT_AT(m.w1, i, j) -= rate*MAT_AT(g.w1, i, j);
        }
    }

    for(size_t i = 0; i < m.b1.rows; i++){
        for(size_t j = 0; j < m.b1.cols; j++){
            MAT_AT(m.b1, i, j) -= rate*MAT_AT(g.b1, i, j);
        }
    }

    for(size_t i = 0; i < m.w2.rows; i++){
        for(size_t j = 0; j < m.w2.cols; j++){
            MAT_AT(m.w2, i, j) -= rate*MAT_AT(g.w2, i, j);
        }
    }

    for(size_t i = 0; i < m.b2.rows; i++){
        for(size_t j = 0; j < m.b2.cols; j++){
            MAT_AT(m.b2, i, j) -= rate*MAT_AT(g.b2, i, j);
        }
    }
}


float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};


void mat_polynomial_transform(Mat y, Mat y_ext){

    for (size_t i = 0; i < y.rows; i++) {
        for (size_t j = 0; j < y_ext.cols; j++) {
            MAT_AT(y_ext, i, j) = pow(MAT_AT(y, i, 0), j+1);
        }
    }

}


int main(void){

    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2
    };

    float yd[] = {
        2,
        2,
        2,
        2
    };

    Mat y = {
        .rows = 4,
        .cols = 1,
        .stride = 1,
        .es = yd
    };

    Mat y_ext = mat_alloc(4, 4);


    mat_polynomial_transform(y, y_ext);

    MAT_PRINT(y);
    MAT_PRINT(y_ext);

    return 0;

    MAT_PRINT(ti);
    MAT_PRINT(to);

    Xor m = xor_alloc();
    Xor g = xor_alloc();

    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.b2, 0, 1);

    float eps = 1e-1;
    float rate = 1e-1;

    printf("cost = %f\n", cost(m, ti, to));
    for (size_t i = 0; i < 50000; i++) {
        finite_diff(m, g, eps, ti, to);
        xor_learn(m, g, rate);
        printf("%zu: cost = %f\n", i, cost(m, ti, to));
    }

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {

            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            forward_xor(m);
            float y = *m.a2.es;

            printf("%zu ^ %zu = %f\n",i , j, y);
        }
    }

    return 0;
}
