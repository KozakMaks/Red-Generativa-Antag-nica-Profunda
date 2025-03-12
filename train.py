import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 512*4*4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = self.main(x)
        return x
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(256*4*4, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 256*4*4)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
num_epochs = 25
for epoch in range(num_epochs):
    for i, (data, _) in enumerate(dataloader):
        real = data.to(device)
        b_size = real.size(0)
        label_real = torch.full((b_size, 1), 1.0, device=device)
        label_fake = torch.full((b_size, 1), 0.0, device=device)
        netD.zero_grad()
        output = netD(real)
        lossD_real = criterion(output, label_real)
        lossD_real.backward()
        noise = torch.randn(b_size, 100, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        lossD_fake = criterion(output, label_fake)
        lossD_fake.backward()
        optimizerD.step()
        netG.zero_grad()
        output = netD(fake)
        lossG = criterion(output, label_real)
        lossG.backward()
        optimizerG.step()
    print('Epoch [{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}'.format(epoch+1, num_epochs, lossD_real.item()+lossD_fake.item(), lossG.item()))
torch.save(netG.state_dict(), 'generator.pth')
