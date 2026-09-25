import torch
import random
import numpy as np
import pickle
import os
import torchvision
import torchvision.transforms as transforms

cpath = os.path.dirname(__file__)

NUM_USER = 500
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = True  # Assuming you want to work with image data
np.random.seed(6)


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        self.data = images
        if normalize:
            self.data = self.data.astype(np.float32) / 255.0
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def data_split(data, num_split):
    """
    Ensure that the data is split evenly among users.
    If data length is not divisible by num_split, discard the remainder.
    """
    delta = len(data) // num_split  # Calculate equal division
    data_lst = [data[i:i + delta] for i in range(0, delta * num_split, delta)]
    return data_lst

def choose_two_digit(split_data_lst):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, 10, replace=False).tolist()
    except:
        print(available_digit)
    return lst

def main():
    # Get CIFAR-10 data
    print('>>> Get CIFAR-10 data.')
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=DATASET_FILE, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATASET_FILE, train=False, download=True, transform=transform)

    train_cifar = ImageDataset(trainset.data, trainset.targets)
    test_cifar = ImageDataset(testset.data, testset.targets)

    cifar_traindata = []
    for number in range(10):
        idx = np.array(train_cifar.target) == number
        cifar_traindata.append(train_cifar.data[idx])
    split_cifar_traindata = []
    for digit in cifar_traindata:
        split_cifar_traindata.append(data_split(digit, 500))

    cifar_testdata = []
    for number in range(10):
        idx = np.array(test_cifar.target) == number
        cifar_testdata.append(test_cifar.data[idx])
    split_cifar_testdata = []
    for digit in cifar_testdata:
        split_cifar_testdata.append(data_split(digit, 1000))


    # Assign train samples to each user
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    print(">>> Data is i.i.d. distributed")

    for user in range(NUM_USER):
        for d in range(10):
            l = len(split_cifar_traindata[d][-1])
            train_X[user] += split_cifar_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()

            l = len(split_cifar_testdata[d][-1])
            test_X[user] += split_cifar_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()
        # indices = list(range(len(train_X[user])))
        # random.shuffle(indices)
        # train_X[user] = [train_X[user][i] for i in indices]
        # train_y[user] = [train_y[user][i] for i in indices]
        #
        # indices = list(range(len(test_X[user])))
        # random.shuffle(indices)
        # test_X[user] = [test_X[user][i] for i in indices]
        # test_y[user] = [test_y[user][i] for i in indices]

    # Setup directory for train/test data
    print('>>> Set data path for CIFAR-10.')
    image = 1 if IMAGE_DATA else 0
    train_path = '{}/data/train/all_data_{}_random_iid.pkl'.format(cpath, image)
    test_path = '{}/data/test/all_data_{}_random_iid.pkl'.format(cpath, image)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in range(NUM_USER):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)

        print('>>> Save data.')


if __name__ == '__main__':
    main()
