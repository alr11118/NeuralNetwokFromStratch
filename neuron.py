x1 = 2
x2 = 3
target = 10

w1 = 1
w2 = 1
b = 0

learning_rate = 0.01

for i in range(10):

    # Forward pass
    prediction = w1 * x1 + w2 * x2 + b

    # Loss
    loss = (prediction - target) ** 2

    # Gradients
    dw1 = 2 * (prediction - target) * x1
    dw2 = 2 * (prediction - target) * x2
    db = 2 * (prediction - target)

    # Update
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db

    print(
        "step:", i,
        "prediction:", prediction,
        "loss:", loss
    )