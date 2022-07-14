import torch
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()
num_image=10
dataset=FakeData(num_image, (1,2,2), 4,to_tensor)
validation=FakeData(num_image, (1,2,2), 4,to_tensor)
#               construct a model with 1 fully connected layer
class NeuralNetwork(torch.nn.Module):
    def __init__(self, image_num_pix: int, num_classes: int):
        """
        Parameters
        ----------
        image_num_pix : number of pixels in image
        num_classes   : number of classes
        """

        # Initialize as any other torch.nn.Module
        super(NeuralNetwork, self).__init__()
        # Define attributes
        # Technical overhead: Save number of pixels to reshape images into vectors
        self.in_features = image_num_pix
        # Define the fully connected layer
        self.lyr1 = torch.nn.Linear(image_num_pix,num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Technical overhead: Used to reshape to avoid RuntimeError when processing 2D data
        img_1 = torch.reshape(img, (img.shape[0], self.in_features))
        # Calculate forward pass using reshape image ("img_1") as input
        output = self.lyr1(img_1)
        return output


# Parameters
num_pix=sum(dataset.image_size[1:])
num_cla=dataset.num_classes
# Question 4.2: Construct model
model = NeuralNetwork(num_pix,num_cla)
# Question 4.3: move model to device
model.to(device)


optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)
test_loader = torch.utils.data.DataLoader(validation,batch_size=2,shuffle=True)


from tqdm import tqdm  # This is used for a fancy training progress bar
max_epochs = 50
# initialize list of losses vs. epochs
loss_list = []
def train(model, train_loader, optimizer, criterion, max_epochs=0):
    for epoch in tqdm(range(max_epochs)):
        #  initialize average loss value
        loss_mean = 0
        # Train 1 Epoch: loop over batches
        for batch_idx, batch in enumerate(train_loader):
            # Train 1 batch
            # organize batch to samples and targets
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

loss_list = train(model,train_loader,optimizer,criterion,max_epochs)


import matplotlib.pyplot as plt

def plot_loss(loss_list):
    plt.figure()
    plt.plot(loss_list)
    plt.title('Loss list with {} number of images'.format(num_image))
    #plt.show()

plot_loss(loss_list)

print(device)