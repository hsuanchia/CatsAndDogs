import torch, logging, os

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
LOSS = torch.nn.CrossEntropyLoss()
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16
MODEL_PATH = './models/'
MODEL_NAME = 'SimpleCNN.pth'
DATA_PATH = './dogs-vs-cats/train/'
TEST_PATH = './dogs-vs-cats/test1/'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', DEVICE)

# 配置 Logger
# logging.basicConfig(level=logging.INFO)  # 配置info
# LOGGER = logging.getLogger(f'{MODEL_NAME}')  #建立一個叫做(f'{MODEL_NAME}')的記錄器
# LOG_FILE = f'{MODEL_PATH}/{MODEL_NAME}.log' #記錄檔案名稱
# file_handler = logging.FileHandler(LOG_FILE)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

class Image_dataset(Dataset):
    def __init__(self, data) -> None:
        self.file_name = data[0]
        self.label = data[1]
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        img_path = os.path.join(DATA_PATH, self.file_name[index])
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
            model.train()
            for imgs, labels in train_DataLoader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                opt.zero_grad()
                loss = LOSS(outputs, labels.to(DEVICE))
                loss.backward()
                opt.step()
                train_avg_loss += loss.item()
                progress.advance(train_batch_tqdm, advance=1)
            print("Start validation...")
            model.eval()
            with torch.no_grad():
                for val_imgs, val_labels in val_DataLoader:
                    val_imgs = val_imgs.to(DEVICE)
                    val_outputs = model(val_imgs)
                    val_loss = LOSS(val_outputs, val_labels.to(DEVICE))
                    val_avg_loss += val_loss.item()
                    progress.advance(val_batch_tqdm, advance=1)

            train_avg_loss /= len(train_DataLoader)
            val_avg_loss /= len(val_DataLoader)

            # LOGGER.info('Epoch {}: train loss: {} |  val loss : {}'.format(num_epochs ,train_avg_loss, val_avg_loss)) #紀錄訓練資訊
            print('Epoch {}: train loss: {} |  val loss : {}'.format(num_epochs ,train_avg_loss, val_avg_loss))
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
    
    train_data, val_data = data_split()
    print(f'Training data: {len(train_data[0])} images')
    print(f'Validation data: {len(val_data[0])} images')

    train_dataset = Image_dataset(train_data)
    train_DataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_dataset = Image_dataset(val_data)
    val_DataLoader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)

    model = SimpleCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

    train(train_DataLoader, val_DataLoader, model, scheduler, opt)