function dx = softmax_grad(x, y)
s = softmax(x);
dx = s - y;