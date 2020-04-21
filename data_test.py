import numpy as np
import h5py



def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def getsamples():
    with h5py.File("./data/full_dataset_vectors.h5", "r") as hf:
        X_train = hf["X_train"][:]
        y_train = hf["y_train"][:]
        X_test = hf["X_test"][:]
        y_test = hf["y_test"][:]


    X_train = np.reshape(np.array(X_train, dtype=np.float32), [-1, 16, 16, 16, 1])
    X_test = np.reshape(np.array(X_test, dtype=np.float32), [-1, 16, 16, 16, 1])
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_test,y_test

def getdata():
    n_lb = 10

    with h5py.File("./data/full_dataset_vectors.h5","r") as hf:

        X_train = hf["X_train"][:]
        y_train = hf["y_train"][:]
        X_test = hf["X_test"][:]
        y_test= hf["y_test"][:]


    X_train = np.reshape(np.array(X_train,dtype=np.float32),[-1,16,16,16,1])
    X_test = np.reshape(np.array(X_test, dtype=np.float32),[-1,16,16,16,1])
    y_train = to_categorical(np.array(y_train),n_lb)
    y_test = to_categorical(np.array(y_test), n_lb)

    return X_train,y_train,X_test,y_test


if __name__=="__main__":
    train_data,train_label,test_data,test_label = getdata()
    print(train_data.shape,train_label.shape)
    print(test_data.shape, test_label.shape)