import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision.models as models
import torchvision.models.video as models
import torch.nn.functional as F

from torchvision.models import swin_t, Swin_T_Weights, resnet50, ResNet50_Weights, efficientnet, EfficientNet_B4_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights, mvit_v2_s, MViT_V2_S_Weights, swin3d_t, Swin3D_T_Weights, r2plus1d_18, R2Plus1D_18_Weights
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np

# Step 1: Load .mat File
def load_mat_data():
    mat0 = scipy.io.loadmat('infrared_frames16x4_train_1.mat')
    mat = scipy.io.loadmat('infrared_frames16x4_train_2.mat')
    # x_train = mat['x1']  # Adjust key names based on your .mat file structure
    x_train = np.concatenate((mat0['x1_1'], mat0['x1_2'], mat0['x1_3'], mat0['x1_4'], mat['x1_1'], mat['x1_2'], mat['x1_3'], mat['x1_4']), axis=4)
    # y_train = mat['y1'].flatten()
    y_train = (np.concatenate((mat0['y1'], mat['y1']), axis=1)).flatten()
    y_train = y_train - 1  # If labels start from 1

    mat = scipy.io.loadmat('infrared_frames16x4_test.mat')
    x_test = mat['x2']
    y_test = (mat['y2']).flatten()
    y_test = y_test - 1

    return x_train, y_train, x_test, y_test

# Step 2: Define Custom Dataset
class MatDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32).permute(4, 2, 3, 0, 1)  # Convert HWCTN → NCTHW
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img, label = self.X[idx], self.y[idx]
        # Resize each frame to (224, 224) using interpolate
        # img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)

        if self.transform:
            img = self.transform(img)  # Apply additional transforms
        return img, label

X_train, y_train, X_test, y_test = load_mat_data()

train_dataset = MatDataset(X_train, y_train)
test_dataset = MatDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model = models.mvit_v2_s(weights=models.MViT_V2_S_Weights.DEFAULT)
num_ftrs = model.head[-1].in_features



num_classes = len(np.unique(y_train))  # Get number of classes from data
# model.fc = nn.Linear(num_ftrs, num_classes)  # Replace FC layer

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),  # Fully connected layer with 512 units
    nn.ReLU(),                 # Activation function
    nn.Dropout(p=0.5),         # Dropout layer with 50% probability
    nn.Linear(512, num_classes)  # Output layer (replace num_classes with actual number)
)

# Step 5: Define Loss, Optimizer, and Device
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 6: Training Function
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        if epoch == 4:  # Epoch 2 (0-based index)
            new_lr = 0.00005
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"⚠️ Learning Rate Changed: Epoch {epoch+1}, New LR = {new_lr}")
            
        elif epoch == 8:  # Epoch 5 (0-based index)
            new_lr = 0.00001
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
                
            print(f"⚠️ Learning Rate Changed: Epoch {epoch+1}, New LR = {new_lr}")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Compute epoch training loss & accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Evaluate test accuracy
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)
        
        test_acc = 100 * correct_test / total_test
        model.train()  # Set back to training mode

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# Step 7: Evaluation Function
def evaluate_model(model, test_loader, criterion=None):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Compute loss (optional)
            if criterion:
                loss = criterion(outputs, labels)
                test_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader) if criterion else None

    # Print results
    print(f"Test Accuracy: {accuracy:.2f}%")
    if criterion:
        print(f"Test Loss: {avg_loss:.4f}")

    return accuracy, avg_loss

# Step 8: Train and Evaluate
train_model(model, train_loader, criterion, optimizer, epochs=12)
evaluate_model(model, test_loader)