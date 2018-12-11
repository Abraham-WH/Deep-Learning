""""
Notation:

    Superscript [l] denotes an object of the lth layer.
        Example: a[4] is the 4th layer activation. W[5] and b[5] are the 5th layer parameters.

    Superscript (i) denotes an object from the ith example.
        Example: x(i) is the ith training example input.

    Lowerscript i denotes the ith entry of a vector.
        Example: a[l]i denotes the ith entry of the activations in layer l, assuming this is a fully connected (FC) layer.

    nH, nW and nC denote respectively the height, width and number of channels of a given layer. If you want to reference a specific layer l, you can also write n[l]H, n[l]W, n[l]C.
    nHprev, nWprev and nCprev denote respectively the height, width and number of channels of the previous layer. If referencing a specific layer l, this could also be denoted n[l−1]H, n[l−1]W, n[l−1]C.
""""
import numpy as np
import h5py
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)

def zero_pad(X, pad):
    """
    输入x是一个四维矩阵（样本数*每个样本像素行数*每个样本像素列数*颜色信道3）
    对第二维和第三维进行padding
    
    np.pad()解释：
    第一个参数是待填充数组
    第二个参数是填充的形状，（2，3）表示前面两个，后面三个
    第三个参数是填充的方法
填充方法：
    constant连续一样的值填充，有关于其填充值的参数。constant_values=（x, y）时前面用x填充，后面用y填充。缺参数是为0000。。。
    edge用边缘值填充
    linear_ramp边缘递减的填充方式
    maximum, mean, median, minimum分别用最大值、均值、中位数和最小值填充
    """
    
    X_pad = np.pad(X,((0,0), (pad,pad), (pad,pad), (0,0)),'constant')
    
    return X_pad

np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1,1] =", x[1,1])
print ("x_pad[1,1] =", x_pad[1,1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])


def conv_single_step(a_slice_prev, W, b):
    """
    fillter矩阵(f*f),与每个局部x进行卷积操作,得到f*f个值a_slice_prev,再把他们相加得到Z
    
   注：1. numpy.sum

    numpy.sum(a, axis=None, dtype=None, out=None, keepdims=<class numpy._globals._NoValue>)[source]

    Sum of array elements over a given axis.
    Parameters:	

    a : array_like

        Elements to sum.

    axis : None or int or tuple of ints, optional

        Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis.

        New in version 1.7.0.

        If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single axis or all the axes as before.

    dtype : dtype, optional

        The type of the returned array and of the accumulator in which the elements are summed. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.

    out : ndarray, optional

        Alternative output array in which to place the result. It must have the same shape as the expected output, but the type of the output values will be cast if necessary.

    keepdims : bool, optional

        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.

        If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes of ndarray, however any non-default value will be. If the sub-classes sum method does not implement keepdims any exceptions will be raised.

    Returns:	

    sum_along_axis : ndarray

        An array with the same shape as a, with the specified axis removed. If a is a 0-d array, or if axis is None, a scalar is returned. If an output array is specified, a reference to out is returned.

    2 np.dot和直接相乘的区别
    2.1. 当为array的时候，默认d*f就是对应元素的乘积，multiply也是对应元素的乘积，dot（d,f）会转化为矩阵的乘积， dot点乘意味着相加，而multiply只是对应元素相乘，不相加

    2.2. 当为mat的时候，默认d*f就是矩阵的乘积，multiply转化为对应元素的乘积，dot（d,f）为矩阵的乘积

    """

    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + b
   
    return Z

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)


def conv_forward(A_prev, W, b, hparameters):
    """
    通过conv_single_step(a_slice_prev, W, b)函数,进行第i个样本的整体卷积操作,前向传播
    四层for循环,核心控制
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
    n_W = int((n_W_prev + 2 * pad - f) / stride + 1)
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:,:]                            # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    vert_start = h * stride             #这四个变量控制fillter的移动,矩阵的起始
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    a_slice_prev =  a_prev_pad[vert_start : vert_end , horiz_start : horiz_end , :]
                    Z[i, h, w, c] = conv_single_step( a_slice_prev , W[:, :, :, c], b[:, :, :, c])
            #结果放在Z矩阵,c是fillter的个数
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =", np.mean(Z))
print("Z[3,2,1] =", Z[3,2,1])
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    池化层
    类似于卷积层
   """
    
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))              
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                  
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_prev_slice = A_prev[i , vert_start : vert_end , horiz_start : horiz_end , c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
 
    cache = (A_prev, hparameters)

    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache

np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)

#后向传播可以用framework自动实现,所以可以optional
