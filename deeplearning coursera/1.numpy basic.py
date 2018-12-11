# GRADED FUNCTION: sigmoid

import numpy as np 

def sigmoid(x):
    s = 1/(1+np.exp(x))
    return s

x = np.array([1, 2, 3])
sigmoid(x)


# GRADED FUNCTION: sigmoid_derivative

def sigmoid_derivative(x):
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    return ds

x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))


# GRADED FUNCTION: image2vector
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return v

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))



# GRADED FUNCTION: normalizeRows

def normalizeRows(x):
    
    x_norm =   x_norm = np.linalg.norm(x,axis=1,keepdims=True)
    print(x_norm)
    # Divide x by its norm.
    x =  x/x_norm
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))


# GRADED FUNCTION: softmax

def softmax(x):
    
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    s = x_exp/x_sum
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
