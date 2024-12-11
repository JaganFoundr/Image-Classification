# Importing Libraries
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transform
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm.auto import tqdm
import random
from timeit import default_timer as timer
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Setting Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creating Dataset
dataset = FashionMNIST(
    root='./data',
    download=True,
    train=True,
    transform=transform.ToTensor(),
    target_transform=None
)

test_dataset = FashionMNIST(
    root='./data',
    download=False,
    train=False,
    transform=transform.ToTensor(),
    target_transform=None
)

# Plotting Dataset Sample
random_index = random.randrange(1, 60001)
images, labels = dataset[random_index]
plt.imshow(images[0], cmap='gray')
plt.axis(False)
plt.show()
print(f"\nThe image label is a {dataset.classes[labels]} and the index number is {labels}")

# Splitting Dataset into Training and Validation

def split(data, valid_percent, seed):
    valid_data = int(data * valid_percent)
    np.random.seed(seed)
    index = np.random.permutation(data)
    return index[valid_data:], index[:valid_data]

training_data, validation_data = split(len(dataset), 0.2, 42)
print("Training data: ", len(training_data))
print("Validation data: ", len(validation_data))
print("\nPortion of the validation data: ", validation_data[:10])

train_sampler = SubsetRandomSampler(training_data)
validation_sampler = SubsetRandomSampler(validation_data)

# Preparing Dataloader
batch_size = 32
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=validation_sampler)

# Defining CNN Model
class FashionCNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )

    def forward(self, x):
        out = self.cnn_block1(x)
        out = self.cnn_block2(out)
        out = self.classifier(out)
        return out

# Creating CNN Model
torch.manual_seed(41)
CNN_model = FashionCNN(input_shape=1, hidden_units=20, output_shape=10).to(device)

# Untrained Prediction
torch.manual_seed(42)
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    prediction = CNN_model(images)
    break
print(prediction[1])

# Accuracy Function
def accuracy(output, labels):
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == labels).item() / len(pred) * 100

# Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(CNN_model.parameters(), lr=0.001)

# Model Run Time Function
def model_run_time(start_time, end_time, device=None):
    total_time = end_time - start_time
    print(f"The model took {total_time:.3f} seconds on device: {device}\n")
    return total_time

# Loss Batch Function
def loss_batch(model, loss_function, images, labels, opt, metrics=accuracy):
    prediction = CNN_model(images)
    loss = loss_function(prediction, labels)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = metrics(prediction, labels) if metrics else None
    return loss.item(), len(images), metric_result

# Evaluation Function
def evaluate(model, loss_function, validation_loader, metrics=accuracy):
    with torch.inference_mode():
        result = [loss_batch(CNN_model, loss_function, images.to(device), labels.to(device), opt=None, metrics=accuracy)
                  for images, labels in validation_loader]

        losses, num, metric = zip(*result)
        total = np.sum(num)
        loss = np.sum(np.multiply(losses, num)) / total

        metric_result = np.sum(np.multiply(metric, num)) / total if metrics else None
        return loss, total, metric_result

# Training and Plotting Function
def train_and_plot(nepochs, model, loss_function, training_loader, validation_loader, opt, metrics=accuracy):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in tqdm(range(nepochs)):
        CNN_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            train_loss, _, train_acc = loss_batch(CNN_model, loss_function, images, labels, opt, metrics=accuracy)

        CNN_model.eval()
        valid_loss, _, valid_acc = evaluate(CNN_model, loss_function, validation_loader, metrics=accuracy)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        print(f"Epoch {epoch + 1}/{nepochs}")
        print(f"Training loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}\n")
        print(f"Training accuracy: {train_acc:.2f}%, Validation accuracy: {valid_acc:.2f}%")
        print("---------------------------------------------------------\n")

    epochs = range(1, nepochs + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Training Loss', color='tab:blue')
    plt.plot(epochs, valid_losses, 'o--', label='Validation Loss', color='tab:red')
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'o-', label='Training Accuracy', color='tab:green')
    plt.plot(epochs, valid_accuracies, 'o--', label='Validation Accuracy', color='tab:orange')
    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    return train_losses, train_accuracies, valid_losses, valid_accuracies

# Training the Model
start_time = timer()
train_losses, train_accuracies, valid_losses, valid_accuracies = train_and_plot(
    2, CNN_model, loss_function, train_loader, validation_loader, opt, metrics=accuracy)
end_time = timer()
model_run_time(start_time, end_time, device=device)

# Installing Additional Libraries
!pip install torchmetrics
!pip install mlxtend

# Confusion Matrix
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

confusion_matrix = ConfusionMatrix(num_classes=len(class_names), task="multiclass").to(device)

all_preds, all_labels = [], []
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

CNN_model.eval()
with torch.inference_mode():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = CNN_model(images)
        _, preds = torch.max(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
conf_matrix_np = confusion_matrix(all_preds, all_labels).cpu().numpy()

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix_np, class_names=class_names, figsize=(8, 8), cmap="Reds")
plt.title("Confusion Matrix")
plt.show()

# Prediction Function
def prediction(images, model):
    input = images.to(device).unsqueeze(0)
    output = CNN_model(input)
    _, pred = torch.max(output, dim=1)
    return pred[0].item()

# Testing the Model
random_index = random.randrange(1, 10001)
images, labels = test_dataset[random_index]
plt.imshow(images[0], cmap='gray')
plt.axis(False)
plt.show()
print("Actual Image Label: ", test_dataset.classes[labels])
print("\nModel Prediction on Testset: ", test_dataset.classes[prediction(images.to(device), CNN_model)])

# Test Set Evaluation
test_loader = DataLoader(dataset=test_dataset, batch_size=200)
test_loss, _, test_acc = evaluate(CNN_model, loss_function, test_loader, metrics=accuracy)
print(f"The testset loss is {test_loss:.4f}.")
print(f"The accuracy of prediction with testset is {test_acc}%.")

# Saving and Loading Model
save_model = FashionCNN(input_shape=1, hidden_units=20, output_shape=10)
torch.save(save_model.state_dict(), "Fashion_convolutional_neural_network.pth")
load_model = save_model
load_model.load_state_dict(torch.load("Fashion_convolutional_neural_network.pth"))
