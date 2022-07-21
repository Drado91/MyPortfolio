from PIL import Image
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision
from torchvision import transforms
"""
def move_files_by_str(source_dir, dest_dir, substr):
    for top, dirs, files in os.walk(source_dir):
        for filename in files:
            if substr not in filename:
                continue
            file_path = os.path.join(top, filename)
            dest_path = os.path.join(dest_dir, filename)
            os.replace(file_path, dest_path)


def count_files_in_dir(dir_path):
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count


def move_files_by_percentage(source_dir, dest_dir, ratio):
    num_files=count_files_in_dir(source_dir)
    num_train_files=int(num_files*ratio)
    for top, dirs, files in os.walk(source_dir):
        for i,filename in enumerate(files):
            label = source_dir.split('//')[-1]
            if i <= num_train_files:
                file_path = os.path.join(source_dir, filename)
                train_dest_path = os.path.join(dest_dir,'Train',label, filename)
                shutil.copyfile(file_path, train_dest_path)
                #os.replace(file_path, train_dest_path)
            else:
                file_path = os.path.join(source_dir, filename)
                train_dest_path = os.path.join(dest_dir,'Test',label, filename)
                shutil.copyfile(file_path, train_dest_path)
                #os.replace(file_path, train_dest_path)
    print(f'{num_train_files} of label {label} files copied to Train, {num_files - num_train_files} to test')



#Split masks from ultrasound images
substr='mask'

source_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\Dataset_BUSI_with_GT//benign"
dest_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\masks//benign"
move_files_by_str(source_dir,dest_dir,substr)

source_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\Dataset_BUSI_with_GT//malignant"
dest_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\masks//malignant"
move_files_by_str(source_dir,dest_dir,substr)

source_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\Dataset_BUSI_with_GT//normal"
dest_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\masks//normal"
move_files_by_str(source_dir,dest_dir,substr)

source_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\Dataset_BUSI_with_GT//benign"
dest_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset//"
move_files_by_percentage(source_dir,dest_dir,0.8)

source_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\Dataset_BUSI_with_GT//malignant"
dest_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset//"
move_files_by_percentage(source_dir,dest_dir,0.8)

source_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\Dataset_BUSI_with_GT//benign"
dest_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset//"
move_files_by_percentage(source_dir,dest_dir,0.8)

source_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset\Dataset_BUSI_with_GT//normal"
dest_dir = "C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset//"
move_files_by_percentage(source_dir,dest_dir,0.8)
"""

def annotations_files_from_folder(root):
    annotations_file = pd.DataFrame()
    for path, subdirs, files in os.walk(root):
        for name in files:
            file_path=os.path.join(path, name)
            dest_path = os.path.join(root, name)
            if 'benign' in name:
                annotations_file=annotations_file.append([[name,1]])
            elif 'normal' in name:
                annotations_file=annotations_file.append([[name,2]])
            elif 'malignant' in name:
                annotations_file=annotations_file.append([[name,3]])
            shutil.copyfile(file_path, dest_path)
    annotations_file_path=root + '/annotations_file.csv'
    annotations_file.reset_index(drop=True, inplace=True)
    annotations_file.to_csv(annotations_file_path,index=False)
    print('annotations file created at: {}'.format(annotations_file_path))
    return annotations_file,annotations_file_path

train_dataset_path="C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset//train"
annotations_file, ann_file_path=annotations_files_from_folder(train_dataset_path)
from PIL import Image
from torch.utils.data import Dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)#read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
training_data = CustomImageDataset(ann_file_path,train_dataset_path,preprocess)
#x=training_data[0]
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataset_path="C:\DAN\MyPortfolio\ResnetUltrasoundClassification\dataset//test"
test_annotations_file, test_ann_file_path=annotations_files_from_folder(test_dataset_path)
test_data = CustomImageDataset(test_ann_file_path,test_dataset_path,preprocess)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
model = torchvision.models.resnet18(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()
from tqdm import tqdm  # This is used for a fancy training progress bar
max_epochs = 50
# initialize list of losses vs. epochs
loss_list = []
def train(model, train_loader, optimizer, criterion, max_epochs=50):
    for epoch in tqdm(range(max_epochs)):
        #  initialize average loss value
        loss_mean = 0
        # Train 1 Epoch: loop over batches
        for batch_idx, batch in enumerate(train_loader):
            # Train 1 batch
            # organize batch to samples and tragets
            samples, targets = batch
            # zero the optimizer gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(samples)
            # Calculate loss
            loss=criterion(outputs, targets)
            # Back-propagation
            loss.backward()
            # optimizer step
            optimizer.step()
            # aggregate loss
            loss_mean += loss.item()
        # Normalize loss
        loss_mean /= (batch_idx + 1)
        # Add loss to list
        loss_list.append(loss_mean)
    return loss_list
loss_list = train(model,train_dataloader,optimizer,criterion,max_epochs)
def plot_loss(loss_list):
    plt.figure()
    plt.plot(loss_list)
    plt.show()
plot_loss(loss_list)
PATH = './dudi_resnet18_50ep.pth'
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))
dataiter = iter(test_dataloader)
images, labels = dataiter.next()
classes = ('benign', 'normal', 'malignant')
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

num_images_to_show=6
imshow(torchvision.utils.make_grid(images[0:num_images_to_show]))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]-1]:5s}' for j in range(num_images_to_show)))

outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]-1]:5s}'
                              for j in range(num_images_to_show)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label-1]] += 1
            total_pred[classes[label-1]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

print('hi')


