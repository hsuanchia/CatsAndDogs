import torch, os, argparse
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, track
from model import SimpleCNN
from data_preprocess import data_split

# Variebles
EPOCHS = 1000
LEARNING_RATE = 1e-4
LOSS = torch.nn.BCEWithLogitsLoss()
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
MODEL_PATH = './models/'
MODEL_NAME = 'SimpleCNN_aug.pth'
TRAIN_PATH = './dogs-vs-cats/train/'
TEST_PATH = './dogs-vs-cats/test1/'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', DEVICE)

class Image_dataset(Dataset):
    def __init__(self, data, data_path, augmentation=False) -> None:
        self.file_name = data[0]
        self.label = data[1]
        self.data_path = data_path
        if(augmentation):
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.file_name[index])
        img = self.transform(Image.open(img_path).convert("RGB"))

        return img, self.label[index]

def train(train_DataLoader, val_DataLoader, model, scheduler, opt):
    best_loss = 888888
    earlystop = 0
    with Progress(TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn()) as progress:
        epoch_tqdm = progress.add_task(description="Epoch progress", total=EPOCHS)
        train_batch_tqdm = progress.add_task(description="Train progress", total=len(train_DataLoader))
        val_batch_tqdm = progress.add_task(description="Val progress", total=len(val_DataLoader))
        for num_epochs in range(EPOCHS):
            train_avg_loss, val_avg_loss = 0, 0
            train_correct, train_total = 0, 0
            val_correct, val_total = 0, 0
            model.train()
            for imgs, labels in train_DataLoader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1).float()
                outputs = model(imgs)
                opt.zero_grad()
                loss = LOSS(outputs, labels)
                loss.backward()
                opt.step()
                train_avg_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).long()
                train_correct += (preds == labels.long()).sum().item()
                train_total += labels.size(0)
                
                progress.advance(train_batch_tqdm, advance=1)
            print("Start validation...")
            model.eval()

            with torch.no_grad():
                for val_imgs, val_labels in val_DataLoader:
                    val_imgs = val_imgs.to(DEVICE)
                    val_labels = val_labels.to(DEVICE).unsqueeze(1).float()
                    val_outputs = model(val_imgs)
                    val_loss = LOSS(val_outputs, val_labels)
                    val_avg_loss += val_loss.item()
                    val_preds = (torch.sigmoid(val_outputs) > 0.5).long()
                    val_correct += (val_preds == val_labels.long()).sum().item()
                    val_total += val_labels.size(0)
                    progress.advance(val_batch_tqdm, advance=1)

            train_avg_loss /= len(train_DataLoader)
            val_avg_loss /= len(val_DataLoader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            print('Epoch {}: train loss: {:.4f} | train acc: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(num_epochs, train_avg_loss, train_acc, val_avg_loss, val_acc))
            scheduler.step(val_avg_loss)
            if best_loss > val_avg_loss:
                print("Model saving...")
                best_loss = val_avg_loss
                path = MODEL_PATH+MODEL_NAME
                torch.save(model.state_dict(), path)
                earlystop = 0
            else:
                earlystop += 1
                print(f"EarlyStop times: {earlystop}")
                if earlystop >= 5:
                    print("Earlystop triggered!")
                    break
            progress.reset(train_batch_tqdm)
            progress.reset(val_batch_tqdm)
            progress.advance(epoch_tqdm, advance=1)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentation or not')
    parser.add_argument('--augmentation', type=bool, default=False, help='Use augmentation or not')
    args = parser.parse_args()
    train_data, val_data = data_split(TRAIN_PATH)
    print(f'Training data: {len(train_data[0])} images')
    print(f'Validation data: {len(val_data[0])} images')

    train_dataset = Image_dataset(train_data, TRAIN_PATH, augmentation=args.augmentation)
    train_DataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_dataset = Image_dataset(val_data, TRAIN_PATH, augmentation=args.augmentation)
    val_DataLoader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)

    model = SimpleCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    train(train_DataLoader, val_DataLoader, model, scheduler, opt)
