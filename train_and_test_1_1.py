from train_and_test_util_1 import *
from my_BCE_loss import *


if CUDA:
    torch.cuda.empty_cache()



catdata = CatDataset(train=True, with_augmentation=False)
trainloader = DataLoader(catdata, batch_size=10, pin_memory=CUDA)
catdata_test = CatDataset(train=False, with_augmentation=False)
testloader = DataLoader(catdata_test, batch_size=1, pin_memory=CUDA)

unet = UNet(in_channels=3, out_channels=1)
if CUDA:
    unet = unet.cuda()

criterion = My_BCELoss()
optimizer = optim.Adam(unet.parameters())
train(unet, criterion, optimizer, trainloader, epoch=10)
test(unet, criterion, testloader, True, "1_1_BCE")

unet = UNet(in_channels=3, out_channels=1)
if CUDA:
    unet = unet.cuda()

criterion = SorensenDiceCoefficientLoss()
optimizer = optim.Adam(unet.parameters())
train(unet, criterion, optimizer, trainloader, epoch=10)
test(unet, criterion, testloader, True, "1_1_Dice")

# torch.save(unet, "unet_1_1.pth")

