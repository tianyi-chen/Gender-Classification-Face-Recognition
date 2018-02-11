from numpy.linalg import norm

# Gradient descent
def grad_descent(f, df, x, y, init_t, alpha, EPS=1e-5, max_iter=30000):
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    cost = []
    iters = []
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            cost.append(f(x, y, t))
            iters.append(iter)
        iter += 1
    return t, cost, iters

