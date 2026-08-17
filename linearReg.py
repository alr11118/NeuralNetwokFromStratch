import sys

#y = 12x - 8
#x = [-2, -1, 0, 1, 2]
#y = [-32, -20, -8, 4, 16]

#y = 2x + 0
#x = [0, 1, 2]
#y = [0, 2, 4]

#y = 21x - 67
#x = [-1, 0, 1]
#y = [-88, -67.1, -46]

#y= 2x + 1
#x = [-1, 0, 1]
#y = [-1, 1, 3]

x = [-2, -1, 0, 1, 2]
y = [2, 0, 0, 2, 6]

w = 1
b = 0
lr = 0.01
loss = sys.maxsize
iterations = 0

while loss > 0.0000001 and iterations < 10000:
    iterations += 1
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

    if(iterations%10 == 0):
        print("w:", w, "b:", b, "loss:", loss)

print("Final w:", round(w, 3))
print("Final b:", round(b, 3))
print("Iterations done:", iterations)
        