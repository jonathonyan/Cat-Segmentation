from train_and_test_util_1 import *


if CUDA:
    torch.cuda.empty_cache()

unet = UNet(in_channels=3, out_channels=1)

catdata = CatDataset(train=True, with_augmentation=True)
trainloader = DataLoader(catdata, batch_size=10, pin_memory=CUDA)
catdata_test = CatDataset(train=False, with_augmentation=False)
testloader = DataLoader(catdata_test, batch_size=1, pin_memory=CUDA)
criterion = SorensenDiceCoefficientLoss()
optimizer = optim.Adam(unet.parameters())#, lr=0.003)
if CUDA:
    unet = unet.cuda()



train(unet, criterion, optimizer, trainloader, epoch=10)
test(unet, criterion, testloader, True, "1_2")

# torch.save(unet, "unet_1_2.pth")
