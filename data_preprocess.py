import os, time
from PIL import Image
from tqdm import tqdm

DATA_PATH = './dogs-vs-cats/train/' # Max image width: 1050, Max image height: 768
TEST_PATH = './dogs-vs-cats/test1/' # Max image width: 500, Max image height: 500


def data_split(ratio=0.8):
    dog_file, cat_file = [], []
    train_img, val_img = [], []
    train_label, val_label = [], []

    for filename in os.listdir(DATA_PATH):
        if filename.startswith('dog'):
            dog_file.append(filename)
        elif filename.startswith('cat'):
            cat_file.append(filename)

    train_img.extend(dog_file[:int(len(dog_file)*ratio)])
    train_label.extend([1]*int(len(dog_file)*ratio))
    val_img.extend(dog_file[int(len(dog_file)*ratio):])
    val_label.extend([1]*(len(dog_file)-int(len(dog_file)*ratio)))
    train_img.extend(cat_file[:int(len(cat_file)*ratio)])  
    train_label.extend([0]*int(len(cat_file)*ratio)) 
    val_img.extend(cat_file[int(len(cat_file)*ratio):])
    val_label.extend([0]*(len(cat_file)-int(len(cat_file)*ratio)))

    return (train_img, train_label), (val_img, val_label)

if __name__ == '__main__':
    train_data, val_data = data_split()
    print(f'Training data: {len(train_data[0])} images')
    print(f'Validation data: {len(val_data[0])} images')
    ### Evaluate max image size in Train / test set ###
    # max_width, max_height = 0, 0
    # for file in tqdm(os.listdir(TEST_PATH)):
    #     img = Image.open(os.path.join(TEST_PATH, file))
    #     img = img.convert("RGB")
    #     # print(f'Image {file} size: {img.size}')
    #     if img.size[0] > max_width:
    #         max_width = img.size[0]
    #     if img.size[1] > max_height:
    #         max_height = img.size[1]
    # print(f'Max image width: {max_width}, Max image height: {max_height}')
        