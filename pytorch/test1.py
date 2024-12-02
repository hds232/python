import torch
import pprint as p
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms as tfs

class Emnist_Model(nn.Module):
    # set the operation mode
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set the image preprocessing method
    DATA_CHANGE = tfs.Compose((
        tfs.ToTensor(),
        tfs.Normalize([0.5], [0.5]),
    ))
    # set up the loss function
    LOSS_F = nn.CrossEntropyLoss().to(DEVICE)
    # dataSet
    trnset = EMNIST(root='./dataset/EMNIST', split='letters', train=True, transform=DATA_CHANGE)
    tstset = EMNIST(root='./dataset/EMNIST', split='letters', train=False, transform=DATA_CHANGE)
    # method
    # Select the computing device according to the actual situation
    def choose(self):
        return self.to(Emnist_Model.DEVICE)
    # initialize the network
    def __init__(self, out_channel):
        super(Emnist_Model, self).__init__()
        # define CNN convolutional neural networks
        # define the convolutional layer
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # define linear layers
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(64*4*4, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, out_channel),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )
    # define the forward propagation process
    def forward(self, x):
        # x = x.view(1, 1, 28, 28)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.linear(x)
        return x
    # train the model
    def start_train(self, learning_rate = 1e-3, epoch = 10, dataloader_batch_size = 100,
                    tensorboard = False, tensor_path = ''):
        # if tensorboard:
        #     if tensor_path == '':
        #         raise ValueError('tensor_path is None')
        #     else:
        #         writer = SummaryWriter(tensor_path)
        # create the optimizer
        OPTIM = optim.SGD(self.parameters(), learning_rate)
        # initialize parameters
        total_running_num = 0
        # model training
        self.train()
        for i in range(epoch):
            # repack the data at the beginning of each iteration
            trn_loader = DataLoader(
                dataset=Emnist_Model.trnset,
                batch_size=dataloader_batch_size,
                shuffle=True,
                drop_last=True
            )
            for data in trn_loader:
                imgs, labels = data
                imgs = imgs.to(Emnist_Model.DEVICE)
                labels = labels.to(Emnist_Model.DEVICE)
                # calculate the output
                output_info = self(imgs)
                # calculate the loss function
                loss = self.LOSS_F(output_info, labels)
                # utilize optimizer backpropagation
                OPTIM.zero_grad()
                loss.backward()
                OPTIM.step()
                # set the print
                total_running_num += 1
                # if tensorboard:
                #     writer.add_scalar('loss', loss, total_running_num)
                if total_running_num % 1000 == 0:
                    print(f'iter_num:{total_running_num}')
        # if tensorboard:
        #     writer.close()
    # test this model
    def start_test(self, dataloader_batch_size = 100):
        # test dataset packaging
        tst_loader = DataLoader(
            dataset=Emnist_Model.tstset,
            batch_size=dataloader_batch_size,
            shuffle=True,
            drop_last=True
        )
        # parameters
        total_num = 0
        # model testing
        self.eval()
        with torch.no_grad() as t_nog:
            for data in tst_loader:
                imgs, labels = data
                imgs = imgs.to(Emnist_Model.DEVICE)
                labels = labels.to(Emnist_Model.DEVICE)
                output_info = self(imgs)
                # calculate the loss function
                loss = self.LOSS_F(output_info, labels)
                predict_labels = output_info.argmax(axis=1)
                # print
                print(f'predicted:{predict_labels}')
                print(f'actual:{labels}')
                print(f'cross entropy:{loss}')
                total_num += torch.sum(predict_labels == labels)
            print(f'success:{total_num / len(Emnist_Model.tstset)}')
    # load models with existing data
    def load_model(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
    # save the trained model
    def save_model(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    obj = Emnist_Model(52).choose()
    obj.load_model('./model/Emnist_Model_200.pth')
    for name, param in obj.named_parameters():
        if param.requires_grad:
            p.pprint(name)
            p.pprint(param.data)
