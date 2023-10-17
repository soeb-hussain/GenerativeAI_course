def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = [] #Store all batches in a list
for files in os.listdir("./cifar-10-batches-py/"):
    if "_batch" in files:
        data.append(extract(os.path.join('./cifar-10-batches-py',files)))
# print(unpickle("data/cifar-10-batches-py/data_batch_1"))
# unpickle("data/cifar-10-batches-py/data_batch_2")
# unpickle("data/cifar-10-batches-py/data_batch_4")
# unpickle("data/cifar-10-batches-py/data_batch_3")
# unpickle("data/cifar-10-batches-py/data_batch_5")