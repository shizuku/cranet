import numpy as np
import pickle

train_images = []
train_labels = []
test_images = []
test_labels = []

with open("train-images-idx3-ubyte", "rb") as f:
    magic = f.read(4)
    size = int.from_bytes(f.read(4), "big")
    r = int.from_bytes(f.read(4), "big")
    c = int.from_bytes(f.read(4), "big")
    for i in range(size):
        mat = []
        for x in range(r):
            mat.append([])
            for y in range(c):
                mat[x].append(int.from_bytes(f.read(1), "big"))
        train_images.append(np.array(mat))

with open("train-labels-idx1-ubyte", "rb") as f:
    magic = f.read(4)
    size = int.from_bytes(f.read(4), "big")
    for i in range(size):
        mat = np.array(int.from_bytes(f.read(1), "big"))
        train_labels.append(mat)

with open("t10k-images-idx3-ubyte", "rb") as f:
    magic = f.read(4)
    size = int.from_bytes(f.read(4), "big")
    r = int.from_bytes(f.read(4), "big")
    c = int.from_bytes(f.read(4), "big")
    for i in range(size):
        mat = []
        for x in range(r):
            mat.append([])
            for y in range(c):
                mat[x].append(int.from_bytes(f.read(1), "big"))
        test_images.append(np.array(mat))

with open("t10k-labels-idx1-ubyte", "rb") as f:
    magic = f.read(4)
    size = int.from_bytes(f.read(4), "big")
    for i in range(size):
        mat = np.array(int.from_bytes(f.read(1), "big"))
        test_labels.append(mat)

mnist = {
    "train_images": train_images,
    "train_labels": train_labels,
    "test_images": test_images,
    "test_labels": test_labels,
}

if __name__ == "__main__":
    with open("mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
