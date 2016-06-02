// -*- c++ -*-
#include <moderngpu/transform.hxx>

struct Crashes {  
    uint var = 0;
    void foo() {
        mgpu::standard_context_t context;

        mgpu::transform(
            [=] MGPU_DEVICE(uint index) {
                printf("boom %d\n", var);
            }, 1, context);

        context.synchronize();
    }
};

struct DoesntCrash {  
    uint var = 0;
    void foo() {
        mgpu::standard_context_t context;
        auto goof = var;
        mgpu::transform(
            [=] MGPU_DEVICE(uint index) {
                printf("hello %d\n", goof);
            }, 1, context);

        context.synchronize();
    }
};

int main(int argc, char** argv) {
    DoesntCrash d;
    d.foo();

    printf("That didn't crash, but this will:\n");
    
    Crashes c;
    c.foo();
    return 0;
}

// nvcc -gencode arch=compute_52,code=sm_52 -std=c++11 -I libs/moderngpu/src --expt-extended-lambda -Xptxas="-v" -lineinfo
