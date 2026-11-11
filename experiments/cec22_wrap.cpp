#include <stdlib.h>
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
#include "cec22_test_func.cpp"
extern "C" void cec22(double *x, double *f, int nx, int mx, int func_num) { cec22_test_func(x, f, nx, mx, func_num); }
