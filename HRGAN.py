import os
import time
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms


print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

img_row = 192
img_col = 704
batch_size = 16
num_epochs = 100
nz = 100
lr = 0.0003

img_transform = transforms.Compose([
    transforms.Resize((img_row, img_col)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    # transforms.Normalize([0.5],[0.5])
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

def loadtraindata():
    path = "./data/"  
    trainset = torchvision.datasets.ImageFolder(
        path, transform=img_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    return trainloader

dataloader = loadtraindata()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convD = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(kernel_size=2, stride=3),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(kernel_size=2, stride=3),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(kernel_size=2, stride=3)
        )
        self.fcD = nn.Sequential(
            nn.Linear(23296, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.convD(input)
        input = input.view(input.size(0), -1)
        input = self.fcD(input)
        output = input.view(input.size(0))
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lnG = nn.Linear(nz, 4 * img_col * img_row)
        self.bnG = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.convG = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(200),
            nn.ReLU(True),
            nn.Conv2d(200, 100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(True),
            nn.Conv2d(100, 50, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.Conv2d(50, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),
            nn.Conv2d(25, 3, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, input):
        input = self.lnG(input)
        input = input.view(input.size(0), 1, 2 * img_row, 2 * img_col)
        input = self.bnG(input)
        output = self.convG(input)
        return output

os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3'
netD = Discriminator()
netG = Generator()
netD = torch.nn.DataParallel(netD.cuda(), device_ids=[0, 1, 2, 3])
netG = torch.nn.DataParallel(netG.cuda(), device_ids=[0, 1, 2, 3])
if torch.cuda.is_available():
    netD = netD.cuda()
    netG = netG.cuda()
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i, (img, _) in enumerate(dataloader):
        nimg = img.size(0)
        rimg = Variable(img).cuda()
        rlabel = Variable(torch.ones(nimg)).cuda()
        flabel = Variable(torch.zeros(nimg)).cuda()
        rout = netD(rimg)
        errD_real = criterion(rout, rlabel)
        noise = Variable(torch.randn(nimg, nz)).cuda()
        fimg = netG(noise)
        fout = netD(fimg)
        errD_fake = criterion(fout, flabel)
        errD = errD_real + errD_fake
        optimizerD.zero_grad()
        errD.backward()
        optimizerD.step()
        noise = Variable(torch.randn(nimg, nz)).cuda()
        fimg = netG(noise)
        output = netD(fimg)
        errG = criterion(output, rlabel)
        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()

        if (i + 1) % 2 == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (epoch, num_epochs, errD.item(), errG.item()))

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
torch.save(netG.state_dict(), './generator111.pth')
torch.save(netD.state_dict(), './discriminator111.pth')
