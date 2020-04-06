from unet import *
from cat_dataset import *
from transfer_dataset import *
from sorensen_dice_coefficient_loss import *
import torch.optim as optim
from tqdm import tqdm
from draw_contour import *

CUDA = torch.cuda.is_available()
EPOCH = 10
BATCH_SIZE = 10
DEBUG = 1

def train(unet, criterion, optimizer, trainloader, epoch=EPOCH):
    for epoch in range(epoch):

        unet.train()

        epoch_loss = 0

        for images, masks in tqdm(trainloader):
            if torch.cuda.is_available():
                images = images.cuda()
                masks = masks.cuda()

            masks_pred = unet(images)

            loss = criterion(masks_pred, masks)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch {}, loss {}".format(epoch, epoch_loss / len(trainloader)))

def test(unet, criterion, testloader, write_mask=False, filename = ""):
    test_loss = 0
    idx=0
    for images, masks in tqdm(testloader):
        if torch.cuda.is_available():
            images = images.cuda()
            masks = masks.cuda()
        masks_pred = unet(images)
        if write_mask:
            write_image(images, masks_pred, masks, idx, filename)
        loss = criterion(masks_pred, masks)

        idx += 1
        test_loss += loss.item()
    print("Test Loss {}".format(test_loss / len(testloader)))
    print("Test Accuracy {}".format(1 - test_loss / len(testloader)))

def write_image(imgs, masks_pred, masks, idx = 0, filename=""):
    masks = masks.cpu().detach().numpy()
    masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
    masks_pred = masks_pred.cpu().detach().numpy()
    masks_pred = masks_pred.reshape(masks_pred.shape[0], masks_pred.shape[2], masks_pred.shape[3])
    cats = np.transpose(imgs.cpu().detach().numpy(), axes=[0,2,3,1])
    for i in range(masks.shape[0]):
        mask_pred = np.copy(masks_pred[i])
        mask_pred[mask_pred < 0.5] = 0
        mask_pred[mask_pred > 0.5] = 1
        mask_pred = mask_pred.astype(np.uint8)
        cv2.imwrite("{}_{}_img.jpg".format(filename, idx), cats[i].astype(np.uint8))
        cv2.imwrite("{}_{}_mask_pred.jpg".format(filename, idx), (mask_pred * 255).astype(np.uint8))
        cv2.imwrite("{}_{}_mask_true.jpg".format(filename, idx), (masks[i] * 255).astype(np.uint8))
        contour = draw_contour(cats[i], masks_pred[i])
        cv2.imwrite("{}_contour_{}.jpg".format(filename, idx), contour)


