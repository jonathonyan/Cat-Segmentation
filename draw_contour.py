import cv2
from cat_dataset import *
from unet import *
from train_and_test_util_1 import *

i = 0

def draw_contour(img, mask):

    mask_cpy = (mask * 255).astype(np.uint8)

    img_cpy = img

    mask_cpy[mask_cpy < 255/2.0] = 0

    mask_cpy[mask_cpy >= 255/2.0] = 255

    edges = cv2.Canny(mask_cpy,50,200)

    img_cpy[np.where(edges != 0)] = np.array([0,255,0])

    return img_cpy

if __name__ == "__main__":
    # Please have the model in the direcory before running it

    unet = UNet(in_channels=3, out_channels=1)


    if CUDA:
        unet = unet.cuda()

    unet.load_state_dict(torch.load("unet_1_3.pt"))

    criterion = SorensenDiceCoefficientLoss()



    catdata_test = CatDataset(train=False, with_augmentation=False)
    testloader = DataLoader(catdata_test, batch_size=1, pin_memory=CUDA)

    test(unet, criterion, testloader, write_mask=True, filename="1_3")
