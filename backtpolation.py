import sys

x = 2
y = 12
lr = 0.01

loss = sys.maxsize
iterations = 0

# Hidden Neuron 
wh = 1
bh = 0

#OutputNeuron 2
wo = 1
bo = 0

while loss > 0.0000001 and iterations < 10000:
    # Set variables
    iterations += 1
    loss = 0

    # Calculate the reasult and error
    zh = x * wh + bh
    ah = max(0, zh)
    ao = ah * wo + bo
    error = ao - y
    loss = error ** 2

    # Adjust weights 
    # Note: d means how much the loss changes with respoect to this thing
    d_ao = 2 * error
    dwo = d_ao * ah
    dbo = d_ao

    dah = d_ao * wo
    dzh = dah * (1 if zh > 0 else 0)
    dwh = dzh * x
    dbh = dzh

    # Update
    wo -= lr * dwo
    bo -= lr * dbo
    wh -= lr * dwh
    bh -= lr * dbh

    if(iterations%10 == 0):
        print("wh:", wh, "bh:", bh, "wo:", wo, "bo:", bo, "loss:", loss)

print("Final wh:", round(wh, 3))
print("Final bh:", round(bh, 3))
print("Final wo:", round(wo, 3))
print("Final bo:", round(bo, 3))
print("prediction:", (wh* x + bh)*wo + bo)
print("Iterations done:", iterations)