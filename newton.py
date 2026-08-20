d_eps = 1e-4  # the approximately epsilon used to calculate the derivative
d_ord = 1e4  # the reciprocal of d_eps
eps = 1e-5  # the stopping criterion of optimize function
tolerance = 1e5  # maximum iteration of optimization loop


def deriv(fun, x):
    """return the derivative of a function at x"""
    return d_ord * (fun(x + d_eps) - fun(x))


def sec_deriv(fun, x):
    """return the second derivative of a function at x"""
    return d_ord * (
        d_ord * (fun(x + 2 * d_eps) - fun(x + d_eps))
        - d_ord * (fun(x + d_eps) - fun(x))
    )


def optimize(start, fun):
    """
    optimization function of newton method
    return a root of the function
    require a starting point x to initiate
    """
    x_prev = start

    temp_sec_deriv = sec_deriv(fun, start)

    if abs(temp_sec_deriv) < 1e-5:
        if abs(deriv(fun, start)) < 1e-5:
            return start
        else:
            print("local minimum does not exist")
            return None
            
    x_curr = start - deriv(fun, start) / sec_deriv(fun, start)
    iter_count = 0
    while abs(x_prev - x_curr) > eps and iter_count < tolerance:
        temp_sec_deriv = sec_deriv(fun, x_curr)
        
        if abs(temp_sec_deriv) < 1e-5:
            if abs(deriv(fun, x_curr)) < 1e-5:
                return x_curr
            else:
                print("local minimum does not exist")
                return None
            
        temp = x_curr - deriv(fun, x_curr) / sec_deriv(fun, x_curr)
        x_prev, x_curr = x_curr, temp
        iter_count += 1

    if iter_count >= tolerance:
        print("local minimum not found within tolerance")
        return None
    
    return x_curr


if __name__ == "__main__":
    print(optimize(3, lambda x: 2))
