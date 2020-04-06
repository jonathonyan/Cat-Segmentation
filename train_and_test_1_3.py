from train_and_test_util_1 import *
from os import listdir

def transfer_learning(unet):

    criterion = SorensenDiceCoefficientLoss()

    optimizer = optim.Adam(unet.parameters())#, lr=0.003)


    transfer_data = TransferDataset()
    transfer_data_loader = DataLoader(transfer_data, shuffle=True, pin_memory=CUDA, batch_size=30)

    train(unet, criterion, optimizer, transfer_data_loader, epoch=20)

    torch.save(unet.state_dict(), "unet_1_3_pretrained.pt")



if CUDA:
    torch.cuda.empty_cache()

input_filenames = set(listdir("."))
filename = "unet_1_3_pretrained.pt"

unet = UNet(in_channels=3, out_channels=1)

if CUDA:
    unet = unet.cuda()


if filename in input_filenames:
    unet.load_state_dict(torch.load("unet_1_3_pretrained.pt"))
else:
    transfer_learning(unet)


catdata = CatDataset(train=True, with_augmentation=False)
trainloader = DataLoader(catdata, batch_size=15, pin_memory=CUDA)
catdata_test = CatDataset(train=False, with_augmentation=False)
testloader = DataLoader(catdata_test, batch_size=1, pin_memory=CUDA)

criterion = SorensenDiceCoefficientLoss()

optimizer = optim.Adam(unet.parameters())#, lr=0.003)

train(unet, criterion, optimizer, trainloader, epoch=10)
test(unet, criterion, testloader, True, "1_3")

torch.save(unet.state_dict(), "unet_1_3.pt")
