d_eps = 0.001 # the approximately epsilon used to calculate the derivative
d_ord = 1000 # basically the reciprocal of d_eps
eps = 0.001 # the stopping criterion of optimize function
max_iter = 1000 # the maximum iteration of optimization loop


def deriv(fun, x):
    """ return the derivative of a function at x """
    return d_ord*(fun(x+d_eps)-fun(x))


def sec_deriv(fun, x):
    """ return the second derivative of a function at x """
    return d_ord*(d_ord*(fun(x+2*d_eps)-fun(x+d_eps))-d_ord*(fun(x+d_eps)-fun(x)))


def optimize(start, fun):
    """ 
    optimization function of newton method
    return a root of the function
    require a starting point x to initiate
    """
    x_prev = start
    x_curr = start - deriv(fun,start)/sec_deriv(fun,start)
    iter_count = 0
    while abs(x_prev-x_curr)>eps and iter_count<max_iter:
        temp = x_curr - deriv(fun,x_curr)/sec_deriv(fun,x_curr)
        x_prev, x_curr = x_curr, temp
        iter_count += 1
    return x_curr


if __name__ == "__main__":
    print(optimize(0, lambda x:(x+2)**3))

