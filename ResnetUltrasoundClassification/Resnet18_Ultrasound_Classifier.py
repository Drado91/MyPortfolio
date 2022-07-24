#!/usr/bin/env python
# coding: utf-8

# ## Ultrasound images classification using ResNet18

# In[95]:


import PIL
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
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')


# ### Data

# Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data reviews the medical images of breast cancer using ultrasound scan. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in classification, detection, and segmentation of breast cancer when combined with machine learning.
# 
# Data
# The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 780 images with an average image size of 500*500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are normal, benign, and malignant.
# 
# https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

# ### Preprocess

# I have downloaded the dataset from kaggle. Dataset was not seperated to train/test so I've created some functions in order to split it by given ratio. Also, data included mask images that I have seperated and used only ultrasound images.
# Ultrasound images in the dataset were given in a different size so I resized and center-cropped it to fit to ResNet18 input size.
# Dataset has different amount of images per class so I checked which class has the minimum number of images ('normal' - 133 imags) and set this number * ratio (106) to be number of images of one class of train set. total number of train set images is (106*3=318). The rest moved for test set. (it makes sense for me that train should have equal number of images per class)

# In[2]:


#Extract dataset
main_path=os.getcwd()
zip_file_path = '{}/archive.zip'.format(main_path)
target_dir = '{}/dataset'.format(main_path)


# In[3]:

"""
with zipfile.ZipFile(zip_file_path) as zf:
     for member in tqdm(zf.infolist(), desc='Extracting '):
        try:
            zf.extract(member, target_dir)
        except zipfile.error as e:
            pass

"""
# In[4]:


#Print dataset directory
def print_dir_hir(path):
    for dirpath, dirnames, filenames in os.walk(path):
        directory_level = dirpath.replace(path, "")
        directory_level = directory_level.count(os.sep)
        indent = " " * 4
        print("{}{}/".format(indent*directory_level, os.path.basename(dirpath)))


# In[5]:


dataset_path=main_path+'/dataset'
print_dir_hir(dataset_path)


# In[59]:


#Few specific functions for dataset directory hierarchy
def move_files_by_str(source_dir,dest_dir,substr):
    for top, dirs, files in  os.walk(source_dir):
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

def move_files_by_percentage(source_dir, dest_dir, ratio,num_files):
    num_train_files=int(num_files*ratio)
    test_samples_cnt=0
    for top, dirs, files in os.walk(source_dir):
        for i,filename in enumerate(files):
            label = source_dir.split('/')[-1]
            if i <= num_train_files:
                file_path = os.path.join(source_dir, filename)
                train_dest_path = os.path.join(dest_dir,'train',label, filename)
                shutil.copyfile(file_path, train_dest_path)
                #os.replace(file_path, train_dest_path)
            else:
                test_samples_cnt+=1
                file_path = os.path.join(source_dir, filename)
                train_dest_path = os.path.join(dest_dir,'test',label, filename)
                shutil.copyfile(file_path, train_dest_path)
                #os.replace(file_path, train_dest_path)
    print(f'{num_train_files} of label {label} files copied to Train, {test_samples_cnt} to test')

def annotations_files_from_folder(root):
    annotations_file = pd.DataFrame()
    for path, subdirs, files in os.walk(root):
        for name in files:
            file_path=os.path.join(path, name)
            dest_path = os.path.join(root, name)
            if 'benign' in name:
                annotations_file=annotations_file.append([[name,0]])
            elif 'normal' in name:
                annotations_file=annotations_file.append([[name,1]])
            elif 'malignant' in name:
                annotations_file=annotations_file.append([[name,2]])
            shutil.copyfile(file_path, dest_path)
    annotations_file_path=root + '/annotations_file.csv'
    annotations_file.reset_index(drop=True, inplace=True)
    annotations_file.to_csv(annotations_file_path,index=False)
    print('annotations file created at: {}'.format(annotations_file_path))
    return annotations_file,annotations_file_path


# In[58]:


def create_dir(path):
    isExist=os.path.isdir(path)
    print(path)
    print('Folder exist = {}'.format(isExist))
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
    return path


# In[76]:


#Define paths:
normal_source_dir = create_dir("{}/dataset/Dataset_BUSI_with_GT/normal".format(main_path))
benign_source_dir = create_dir("{}/dataset/Dataset_BUSI_with_GT/benign".format(main_path))
malignant_source_dir = create_dir("{}/dataset/Dataset_BUSI_with_GT/malignant".format(main_path))

normal_mask_dest_dir = create_dir("{}/dataset/masks/normal".format(main_path))
benign_mask_dest_dir = create_dir("{}/dataset/masks/benign".format(main_path))
malignant_mask_dest_dir = create_dir("{}/dataset/masks/malignant".format(main_path))

dest_dir = create_dir("{}/dataset/".format(main_path))
train_dataset_path=create_dir(os.path.join(dest_dir,'train'))
test_dataset_path=create_dir(os.path.join(dest_dir,'test'))                         

create_dir("{}/dataset/train/normal".format(main_path))
create_dir("{}/dataset/train/benign".format(main_path))
create_dir("{}/dataset/train/malignant".format(main_path))

create_dir("{}/dataset/test/normal".format(main_path))
create_dir("{}/dataset/test/benign".format(main_path))
create_dir("{}/dataset/test/malignant".format(main_path))


# In[28]:


#Split masks from ultrasound images
"""
substr='mask'
move_files_by_str(normal_source_dir,normal_mask_dest_dir,substr)
move_files_by_str(benign_source_dir,benign_mask_dest_dir,substr)
move_files_by_str(malignant_source_dir,malignant_mask_dest_dir,substr)
"""

# In[29]:


#Find minimum number of images per class
dirs=[normal_source_dir,benign_source_dir,malignant_source_dir]
find_min=np.inf
for dir in dirs:
    num_files=count_files_in_dir(dir)
    print(num_files)
    if num_files < find_min:
        find_min=num_files
print('minimum samples in one class: {}'.format(find_min))


# In[62]:


#Split dataset to train and test
"""
move_files_by_percentage(normal_source_dir,dest_dir,0.8,find_min)
move_files_by_percentage(benign_source_dir,dest_dir,0.8,find_min)
move_files_by_percentage(malignant_source_dir,dest_dir,0.8,find_min)
"""

# In[63]:


#Create annotations csv file for train
#train_annotations_file, train_annotations_file_path=annotations_files_from_folder(train_dataset_path)
train_annotations_file_path="/Users/drado/GitCode/Portfolio/ResnetUltrasoundClassification/dataset/train/annotations_file.csv"
#Create annotations csv file for test
#test_annotations_file, test_annotations_file_path=annotations_files_from_folder(test_dataset_path)
test_annotations_file_path="/Users/drado/GitCode/Portfolio/ResnetUltrasoundClassification/dataset/test/annotations_file.csv"

# In[64]:


#Print dataset directory after some preprocess
dataset_path=main_path+'/dataset'
print_dir_hir(dataset_path)


# In[65]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[66]:


#Create dataset class (I've downloaded the dataset from kaggle not from torchvision)
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


# In[164]:


#Define transform
preprocess = transforms.Compose([
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomRotation(8, resample=PIL.Image.BILINEAR),
    transforms.Resize((224,224)),
    #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),

])

#Define model parameters
batch_size=5


# ### Model - ResNet18 pretrained

# In[165]:


classes = ('benign', 'normal', 'malignant')


# In[166]:


#Load train dataset
train_data = CustomImageDataset(train_annotations_file_path,train_dataset_path,preprocess)
train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)


# In[176]:


#Training-validate split
from torch.utils.data import DataLoader, random_split
num_train=train_data.__len__()
tra_val_ratio=0.85
train,validation = random_split(train_data,[int(num_train*tra_val_ratio),num_train-int(num_train*tra_val_ratio)])

# In[179]:


train_dataloader = DataLoader(train, batch_size=batch_size,shuffle=True)
validation_dataloader = DataLoader(validation, batch_size=batch_size,shuffle=True)


# In[184]:


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=[15, 15])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_dataset(dataset, n=6):
    img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))for i in range(len(dataset))))
    plt.imshow(img)
    plt.axis('off')


# In[185]:


#Show train images and their label
num_images_to_show=batch_size
train_dataloader_to_show = DataLoader(train_data, batch_size=num_images_to_show,shuffle=True)
dataiter = iter(train_dataloader_to_show)
images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid(images[0:num_images_to_show]))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(num_images_to_show)))
print('Shape of image is {}'.format(images[0].shape)) 
print('Amount of Images in Train Dataset is {}'.format(train_dataloader.sampler.num_samples))
print('Amount of Images in Validation Dataset is {}'.format(validation_dataloader.sampler.num_samples))


# In[186]:


#Load ResNet18
model = torchvision.models.resnet18(pretrained=True)


# In[187]:


#Define optimizer, loss function and max epochs
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
#ptimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 200


# In[191]:


from tqdm import tqdm 
from torchmetrics.functional import accuracy

# initialize list of losses vs. epochs
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

def train(model, train_loader, validloader, optimizer, criterion, max_epochs=50):
    epoch_num=1
    for epoch in tqdm(range(max_epochs)):
        #  initialize average loss value
        train_loss_mean = 0
        acc=0
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
            train_loss_mean += loss.item()
            #acc+=accuracy(outputs, targets)

        # Normalize loss
        train_loss_mean /= (batch_idx + 1)
        #train_acc = torch.sum(outputs == targets)
        
        valid_loss_mean = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in validloader:
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target,labels)
            # Calculate Loss
            valid_loss_mean += loss.item()

        #val_acc = torch.sum(labels == target)

        print(f'Epoch {epoch_num} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss_mean / len(validloader)}')
        print('Loss: {}, Epoch: {}'.format(loss_mean, epoch_num))
        epoch_num+=1
        # Add loss to list
        train_loss_list.append(loss_mean)
        train_acc_list.append(train_acc)
    return loss_list
loss_list = train(model,train_dataloader,validation_dataloader,optimizer,criterion,max_epochs)


# In[660]:


import matplotlib.pyplot as plt
import math
def plot_loss(loss_list):
    epochs_axis=np.arange(len(loss_list))+1
    xticks = range(math.floor(min(epochs_axis)), math.ceil(max(epochs_axis))+1,int(len(loss_list)/5))
    plt.figure()
    plt.plot(epochs_axis,loss_list)
    plt.title('Loss Curve - {} Epochs'.format(len(epochs_axis)))
    plt.xticks(xticks)
    plt.show()

plot_loss(loss_list)


# In[661]:


PATH = './dudi_resnet18_50ep_1e-1lr_b.pth'
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))


# In[662]:


#Load test dataset
test_data = CustomImageDataset(test_ann_file_path,test_dataset_path,preprocess)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# In[678]:


dataiter = iter(test_dataloader)
images, labels = dataiter.next()


# In[680]:


#Show test images and their label
num_images_to_show=5
imshow(torchvision.utils.make_grid(images[0:num_images_to_show]))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(num_images_to_show)))


# In[681]:


outputs = model(images)


# In[682]:


#Print test images predictions
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(5)))


# In[683]:


#Accuracy of total train images
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in train_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the train images: {100 * correct // total} %')


# In[684]:


#Accuracy of total test images
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

print(f'Accuracy of the network on the test images: {100 * correct // total} %')


# In[685]:


#Accuracy of train images by class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in train_dataloader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label-1]] += 1
            total_pred[classes[label-1]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy of train for class: {classname:5s} is {accuracy:.1f} %')


# In[686]:


#Accuracy of test images by class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

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

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy of test for class: {classname:5s} is {accuracy:.1f} %')


# In[687]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in test_dataloader:
        output = model(inputs) # Feed Network
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        



# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
#plt.savefig('output.png')


# Intermediate Summary:
# Net results are not good enough, it seems from confusion matrix that model is biased to classify to benign label. Here are some possible issues and solutions.
# 
# | Possible issue | Possible solution |
# | --- | --- |
# | Lack of data | Download more data. Change train to test ratio and check again. |
# | --- | --- |
# | Transform Parameters | Change Resize method. Change normalize value (mean,std.) |
# | --- | --- |
# | Hyper Parameters | Change learning rate, batch size, loss function, optimizer. |
# | --- | --- |
# | Classes similarity | Train and test the model for normal and malignant (best and worst case). Without benign (middle case). |
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
