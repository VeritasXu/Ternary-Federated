import os
import sys
import shutil
import random
import os.path
import torchvision
import pandas as pd
from PIL import Image
from skimage import io
from utils.config import Args
import torch.utils.data as read_d
from torchvision import transforms
from torch.utils.data import Dataset



path1 = sys.path[1]
# print(path1)

###MNIST
mnist_root = path1+'/data'
mnist_root_img = mnist_root+'/MNIST/raw'
mnist_save_path = path1+'/data/MNIST'


mnist_train = torchvision.datasets.MNIST(root=mnist_root, train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)

mnist_test = torchvision.datasets.MNIST(root=mnist_root, train=False,
                                      transform=torchvision.transforms.ToTensor(),
                                      download=False)


def init_class(n_client, nc):
    """
    Used for non-iid setting, initialize classes in each client
    :param n_client:
    :param nc:
    :return:
    """
    n_part = [[] for _ in range(n_client)]
    all_class = [_c for _c in range(10)]
    assigned_set = set([])

    for _i in range(0, n_client):
        while len(set(n_part[_i])) < nc:
            tmp_c = random.choice(all_class)
            if tmp_c not in n_part[_i]:
                n_part[_i].append(tmp_c)

        assigned_set = assigned_set | set(n_part[_i])

    return n_part, assigned_set



def assign_class(n_client, nc):
    """
    Used for non-iid settings, after init_class(N_Client, Nc), if any class is missing, re-assign
    :param n_client:
    :param nc:
    :return:
    """
    n_part = [[] for _ in range(n_client)]
    all_class = [_c for _c in range(10)]
    assigned_set = set([])

    while assigned_set != set(all_class):
        n_part, assigned_set = init_class(n_client, nc)

    return n_part


def default_loader(path):
    """
    define loader
    :param path:
    :return:
    """
    return Image.open(path).convert("L")


class MyDataset(Dataset):
    def __init__(self, part_path, part_name, transform=None, target_transform=None, loader=default_loader):
        self.images = part_name
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.pth = part_path

    def __getitem__(self, index):
        name, label = self.images[index]
        pic_path = self.pth+'/'+str(name)+'.jpg'
        img = self.loader(pic_path)
        # print(img.size)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)



def MNIST_to_IMG(save_path, train_set):
    """
    To change binary data to images
    :param save_path:
    :param train_set:
    :param test_set:
    :param train:
    :return:
    """
    data_path = save_path + '/train/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
        img_path = data_path + str(i) + '.jpg'
        io.imsave(img_path, img.numpy())





def mnist_img_resize(img_file, path_save, width,height):
    """
    resize the image
    :param img_file: image root
    :param path_save: save path
    :param width:
    :param height:
    :return:
    """
    img = Image.open(img_file)
    new_image = img.resize((width, height), Image.ANTIALIAS)
    new_image.save(os.path.join(path_save, os.path.basename(img_file)))


def split_mnist(args):
    """
    main, split mnist according to config
    :param args:
    :param To_Img:
    :param RESIZE:
    :return:
    """
    with open(path1+'/data/data_config.txt', 'w') as f:
        f.writelines(str(Args.num_C)+str(Args.Nc)+str(Args.iid))

    # client number
    num_client = args.num_C
    # Nc
    nc = args.Nc

    train_loader = read_d.DataLoader(mnist_train, batch_size=1)


    train_part = []
    for i, (x, y) in enumerate(train_loader):
        train_part.append((i, int(y.data.numpy())))



    for _ in range(num_client):
        # if dir not exist, creat it
        if os.path.exists(mnist_save_path + '/part' + str(_) + '/') is False:
            os.mkdir(mnist_save_path + '/part' + str(_) + '/')
        else:
            shutil.rmtree(mnist_save_path + '/part' + str(_) + '/')
            os.mkdir(mnist_save_path + '/part' + str(_) + '/')

    train_set = (
        torchvision.datasets.mnist.read_image_file(os.path.join(mnist_root_img, 'train-images-idx3-ubyte')),
        torchvision.datasets.mnist.read_label_file(os.path.join(mnist_root_img, 'train-labels-idx1-ubyte'))
    )

    MNIST_to_IMG(mnist_save_path, train_set)

    if args.iid is True:
        # iid
        random.shuffle(train_part)
        num_samp = int(60000 / num_client)
        part = [train_part[_ * num_samp:(_ + 1) * num_samp] for _ in range(num_client)]
        # for i in range(3):
        #     part[i] = train_part[i * 1800: (i+1) * 1800]



    else:
        # non-iid
        n_label = assign_class(num_client, nc)
        num_sample = int(60000/num_client)
        part = [[] for _ in range(num_client)]

        for j in range(num_client):
            sample_in = 0
            #shuffle trainset
            random.shuffle(train_part)
            for idx, y in train_part:
                if sample_in < num_sample and y in n_label[j]:
                    part[j].append((idx, y))
                    sample_in += 1



    for _ in range(num_client):
        for _i in part[_]:
            mnist_img_resize(mnist_save_path + '/train/' + str(_i[0]) + '.jpg', mnist_save_path + '/part' + str(_), 28, 28)

        part_pd = pd.DataFrame(part[_])
        part_pd.to_csv(mnist_save_path + '/part' + str(_) + '.csv', index=False, header=False)



with open(path1+'/data/data_config.txt', 'r') as txt:
    # read all lines
    content = txt.readlines()
    txt.close()


if str(Args.num_C)+str(Args.Nc)+str(Args.iid) != str(content[0]):
    print('Re-divide MNIST')
    split_mnist(Args)
else:
    print('Previous divided MNIST')

M_part = locals()
for c in range(Args.num_C):
    tmp = pd.read_csv(mnist_save_path+'/part' + str(c) + '.csv', header=None).values
    M_part[str(c)] = MyDataset(mnist_save_path + '/part' + str(c), tmp, transform=transforms.ToTensor())

