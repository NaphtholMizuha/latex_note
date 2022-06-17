import copy

import torch.utils.data
import torch.nn as nn
import torchvision

import trans
import torchvision.datasets as datasets
import torch

import matplotlib.pyplot as plt
import numpy as np
from cnn import CNN

train_path = 'data/train'
test_path = 'data/test'
plt.rcParams['font.sans-serif'] = 'Songti SC'

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    train_dataset = datasets.ImageFolder(train_path, trans.transformer['train'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataset = datasets.ImageFolder(test_path, trans.transformer['test'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    # parameters
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    num_epoches = 24

    # Get a batch of training data
    inputs, classes = next(iter(train_dataloader))
    class_names = train_dataset.classes
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    epoch_losses = []
    test_accuracies = []

    model.train()
    for epoch in range(num_epoches):
        print(f'Epoch {epoch + 1} / {num_epoches}')
        # Train
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # Forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / train_size
        epoch_losses.append(epoch_loss)
        epoch_accuracy = float(running_corrects) / train_size
        print(f'Train Set: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}')

        # Test
        test_loss = 0.0
        test_correct = 0
        for inputs, labels in test_dataloader:
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            test_correct += torch.sum(preds == labels.data) / inputs.size(0)
        test_loss = test_loss / test_size
        test_accuracy = float(test_correct) / test_size
        test_accuracies.append(test_accuracy)
        print(f'Test Set: Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.4f}')
        print('-' * 20)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())
    print(f'Best Accuracy: {test_accuracy}')
    model.load_state_dict(best_model_weights)

    plt.plot(range(len(epoch_losses)), epoch_losses)
    plt.title('训练集损失函数随epoch下降情况')
    plt.savefig('train-loss.pdf')
    plt.show()

    plt.plot(range(len(test_accuracies)), test_accuracies)
    plt.title('测试集准确率函数随epoch上升情况')
    plt.savefig('test-acc.pdf')
    plt.show()


