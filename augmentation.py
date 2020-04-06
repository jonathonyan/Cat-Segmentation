import cv2
from cat_dataset import *




def flip(cat, mask):
    flip_direction = np.random.randint(3) - 1
    cat_fliped = cv2.flip(cat, flip_direction)
    mask_fliped = cv2.flip(mask, flip_direction)
    return cat_fliped, mask_fliped


def rotate(cat, mask):
    rotate_degree = np.random.randint(-90, 90)

    centre = np.array([cat.shape[0], cat.shape[1]]) / 2.0
    rot = cv2.getRotationMatrix2D(tuple(centre), rotate_degree,1.0)

    cat_rotated = cv2.warpAffine(cat, rot, (cat.shape[1], cat.shape[0]),flags=cv2.INTER_LINEAR)
    mask_rotated = cv2.warpAffine(mask, rot, (cat.shape[1], cat.shape[0]) ,flags=cv2.INTER_LINEAR)

    return cat_rotated, mask_rotated

def shift(cat, mask):
    shift_unit = (np.random.randint(-cat.shape[0]//4, cat.shape[0]//4),\
                                    np.random.randint(-cat.shape[1]//4, cat.shape[1]//4) )
    M = np.float32([[1,0,shift_unit[0]], [0,1,shift_unit[1]]])
    cat_shifted = cv2.warpAffine(cat, M, (cat.shape[1], cat.shape[0]))
    mask_shifted = cv2.warpAffine(mask, M, (cat.shape[1], cat.shape[0]))
    return cat_shifted, mask_shifted


def change_saturation(cat, mask):
    hsv = cv2.cvtColor(cat, cv2.COLOR_BGR2HSV).astype(np.float64)

    hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    cat_changed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return cat_changed, mask


AUGMENTATION_FUNCTIONS = [flip, rotate, shift, change_saturation]


if __name__ == "__main__":
    #Try some augmentation
    folder_path = TRAIN_DATASET_PATH
    input_path = join(folder_path, "input")
    mask_path = join(folder_path, "mask")

    input_filename = listdir(input_path)[0]

    data_id = input_filename.split(".")[1]

    cat_filename = join(input_path, input_filename)
    mask_filename = join(mask_path, "mask_cat.{}.jpg".format(data_id))

    cat = cv2.imread(cat_filename, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)


    cv2.imwrite("cat.jpg", cat)
    cv2.imwrite("mask.jpg", mask)


    cat_fliped, mask_fliped = flip(cat, mask)

    cv2.imwrite("cat_fliped.jpg", cat_fliped)

    cv2.imwrite("mask_fliped.jpg", mask_fliped)



    cat_rotated, mask_rotated = rotate(cat, mask)

    cv2.imwrite("cat_rotated.jpg", cat_rotated)

    cv2.imwrite("mask_rotated.jpg", mask_rotated)


    # cat_zoomed, mask_zoomed = zoom(cat, mask)

    # cv2.imwrite("cat_zoomed.jpg", cat_zoomed)

    # cv2.imwrite("mask_zoomed.jpg", mask_zoomed)


    cat_shifted, mask_shifted = shift(cat, mask)

    cv2.imwrite("cat_shifted.jpg", cat_shifted)

    cv2.imwrite("mask_shifted.jpg", mask_shifted)


    cat_changed, mask_changed = change_brightness(cat, mask)

    cv2.imwrite("cat_changed.jpg", cat_changed)

    cv2.imwrite("mask_changed.jpg", mask_changed)




