import torch, os, time, csv, argparse
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from model import SimpleCNN

TEST_PATH = './dogs-vs-cats/test1/'
TEST_LABEL_PATH = './dogs-vs-cats/test1_label.csv'
IMAGE_SIZE = (512, 512)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', DEVICE)

class Image_dataset_inference(Dataset):
    def __init__(self, data, data_path, label_path) -> None:
        self.file_name = data
        self.data_path = data_path
        self.label_path = label_path
        self.file_label = {}
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        self.load_from_csv()

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.file_name[index])
        img = self.transform(Image.open(img_path).convert("RGB"))

        return img, self.file_label.get(os.path.splitext(self.file_name[index])[0], -1)

    def load_from_csv(self):
        """Read labels from CSV file"""
        if not os.path.exists(self.label_path):
            print(f"CSV file isn't exist: {self.label_path}")
            return
        try:
            with open(self.label_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        self.file_label[row['id']] = int(row['label'])
                    except (ValueError, KeyError) as e:
                        print(f"Error on read row: {e}, row={row}")
                        continue
            print(f"Read csv success, has read {len(self.file_label)} labels")
        except Exception as e:
            print(f"Read csv failed: {e}")

def inference(test_DataLoader, model):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC curve
    
    with torch.no_grad():
        for imgs, labels in track(test_DataLoader):
            outputs = model(imgs.to(DEVICE))
            probs = torch.sigmoid(outputs).squeeze()  # Probability of positive class
            preds = (probs > 0.5).long()  # Convert to class 0 or 1
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 1. Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {acc:.4f}')
    
    # 2. Precision & Recall
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Confusion Matrix:\n{cm}')
    plt.figure(figsize=(2,2))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks([i for i in range(2)])
    plt.yticks([i for i in range(2)])

    thresh = cm.max()/2
    for i in range(2):
        for j in range(2):
            plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j] > thresh else "black")
    plt.xlabel("True label")
    plt.ylabel("Predict label")
    plt.tight_layout()
    plt.savefig('./cnf_mat.png', dpi=300, bbox_inches='tight')
    print('Confusion matrix saved to ./cnf_mat.png')
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('./roc_curve.png', dpi=300, bbox_inches='tight')
    print(f'ROC AUC: {roc_auc:.4f}')
    print('ROC curve saved to ./roc_curve.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for Cats vs Dogs Classification')
    parser.add_argument('--model_path', type=str, default='./models/SimpleCNN_30ep.pth', help='Path of trained model')
    args = parser.parse_args()
    test_data = [x for x in os.listdir(TEST_PATH)]

    test_dataset = Image_dataset_inference(test_data, TEST_PATH, label_path=TEST_LABEL_PATH)
    test_DataLoader = DataLoader(test_dataset, shuffle=False, batch_size=16)

    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path))

    inference(test_DataLoader, model)
    

