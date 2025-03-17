function y = softmax(x)
x_max = max(x, [], 2);
y = exp(x - x_max) ./ sum(exp(x - x_max), 2);