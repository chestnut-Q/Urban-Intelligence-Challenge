from dataset import mydataset
import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from torch.autograd import Variable # torch 中 Variable 模块

def main(config):
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = mydataset(transform=transform_train)
    # if you want to add the addition set and validation set to train
    # train_dataset = tiny_caltech35(transform=transform_train, used_data=['train', 'val', 'addition'])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    val_dataset = mydataset(transform=transform_test, used_data='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    test_dataset = mydataset(transform=transform_test, used_data='val')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    # model = base_model(class_num=config.class_num)
    # use vgg19
    # model = torchvision.models.vgg19(pretrained=True)
    model = torchvision.models.resnet18(pretrained=True)
    if torch.cuda.is_available():
        print('Use cuda')
        model = model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    creiteron = torch.nn.CrossEntropyLoss()

    # you may need train_numbers and train_losses to visualize something
    train_numbers, train_losses, accuracies = train(config, train_loader, model, optimizer, scheduler, creiteron, val_loader, test_loader)

    # you can use validation dataset to adjust hyper-parameters
    val_accuracy = test(val_loader, model)
    test_accuracy = test(test_loader, model)
    # test_accuracy = 0
    print('===========================')
    print("val accuracy:{}%".format(val_accuracy * 100))
    print("test accuracy:{}%".format(test_accuracy * 100))
    return val_accuracy.cpu(), test_accuracy.cpu()


def train(config, data_loader, model, optimizer, scheduler, creiteron, val_loader, test_loader):
    model.train()
    train_losses = []
    train_numbers = []
    accuracies = []
    counter = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, label) in enumerate(data_loader):
            # use cuda
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()

            data = data.to(torch.float32)
            # data = Variable(torch.unsqueeze(data, dim=0).float(), requires_grad=False)
            output = model(data)
            # print(output.shape)
            loss = creiteron(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            accuracy = (label == output.argmax(dim=1)).sum() * 1.0 / output.shape[0]
            if batch_idx % 5 == 0:
                print('Train Epoch: {} / {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch, config.epochs, batch_idx * len(data), len(data_loader.dataset),
                                          100. * batch_idx / len(data_loader), loss.item(), accuracy.item()))
                train_losses.append(loss.item())
                train_numbers.append(counter)
                accuracies.append(accuracy.item())
        scheduler.step()
        torch.save(model.state_dict(), './myres_model.pth')

        if epoch % 5 == 0:
            test_model = torchvision.models.resnet18()
            test_model.load_state_dict(torch.load('./myres_model.pth'))
            test_model.cuda()
            val_accuracy = test(val_loader, test_model)
            test_accuracy = test(test_loader, test_model)
            print("val accuracy:{}%".format(val_accuracy * 100))
            print("test accuracy:{}%".format(test_accuracy * 100))

    return train_numbers, train_losses, accuracies


def test(data_loader, model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in data_loader:
            # use cuda
            
            data = data.cuda()
            data = data.to(torch.float32)
            label = label.cuda()

            output= model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum()
    accuracy = correct * 1.0 / len(data_loader.dataset)
    return accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[200, 176])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)  
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 50])

    config = parser.parse_args()
    main(config)


# 0.00005, 