import sys

x = [-2, -1, 0, 1, 2]
y = [-32, -20, -8, 4, 16]

w = 1
b = 0
lr = 0.01
loss = sys.maxsize

while loss > 0.01:
    loss = 0
    dw = 0
    db = 0

    for i in range(len(x)):
        prediction = x[i] * w + b
        error = prediction - y[i]

        loss += error ** 2
        dw += 2 * error * x[i]
        db += 2 * error

    loss /= len(x)
    dw /= len(x)
    db /= len(x)

    w = w - lr * dw
    b = b - lr * db

    print("w:", w, "b:", b, "loss:", loss)

print("Final w:", w)
print("Final b:", b)
        