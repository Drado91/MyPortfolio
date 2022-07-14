In this excercise, you will implement a 1-layer fully connected network to map random 2D arrays of noise to randomly assigned classes (AKA labels or targets).
The questions appear in increasing difficulty levels.
We will use the PyTorch library.

## The Excercise
Follow the instructions in the following boxes
Lines that contain "pass" commands and variables set to "None" should be replaced by your answers


# Question 1  : Write import statements for:
#          1.1: the "torch" module
#          1.2: the "torchvision.transforms" module (name it "transforms")
#          1.3: the "FakeData" class of the "torchvision.datasets" module
#          1.4: the class used for loading data into neural networks in "PyTorch"
pass
pass
pass
pass

# Question 2.1: check for cuda (GPU usage) availability
pass
# Question 2.2: define the device ("torch.device" class instance) used for storing training data and models
# Requirements: device type should depend on cuda availability!
pass

# Question 3.1: Create a fake dataset using the "FakeData" class
# Requirements: dataset should be a total 4 images, image's shape (1,2,2) and 4 classes
#               When initializing an instance of "FakeData", set its "transform" argument to be the "to_tensor" callable
to_tensor = transforms.ToTensor()
pass

# Question 4.1: Define a linear model:
# Requirements: define a class that inherits from torch.nn.Module
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
        self.lyr1 = None

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Technical overhead: Used to reshape to avoid RuntimeError when processing 2D data
        img_1 = torch.reshape(img, (img.shape[0], self.in_features))
        # Calculate forward pass using reshape image ("img_1") as input
        output = None
        return output


# Parameters
pass
# Question 4.2: Construct model
model = NeuralNetwork()
# Question 4.3: move model to device
pass

# Question 5.1: define optimizer and loss
# Requirements: optimizer should be Adam, loss should be cross entropy
#               both should be callables. implementations are available in PyTorch and can be used here, like we did with "FakeData"
#               set the optimizer's learning rate to be 1e-2
optimizer = None
criterion = None

# Question 6.1: Create an Instance of the class mentioned in Q1.4 for loading the data in batches
# Requirements: set batch size to be 2
#               data should be shuffled during each epoch
train_loader = None

# Question 7.1: Write a training loop for 50 epochs
# Question 7.2: insert the inputs to the device in the for loop.
from tqdm import tqdm  # This is used for a fancy training progress bar


max_epochs = 500
# initialize list of losses vs. epochs
loss_list = []
def train(model, train_loader, optimizer, criterion, max_epochs=0):
    for epoch in tqdm(range(max_epochs)):
        #  initialize average loss value
        loss_mean = 0
        # Train 1 Epoch: loop over batches
        for batch_idx, batch in enumerate(train_loader):
            # Train 1 batch
            # organize batch to samples and tragets
            pass
            # zero the optimizer gradients
            pass
            # Forward pass
            pass
            # Calculate loss
            pass
            # Back-propagation
            pass
            # optimizer step
            pass
            # aggregate loss
            loss_mean += loss.item()
        # Normalize loss
        loss_mean /= (batch_idx + 1)
        # Add loss to list
        loss_list.append(loss_mean)
    return loss_list

loss_list = train()

# Question 8.1: plot loss during training
import matplotlib.pyplot as plt

def plot_loss(loss_list):
    plt.figure()
    plt.plot(loss_list)
    plt.show()

plot_loss(loss_list)


"""
## Questions
1. What is the effect of different number of epochs? different learning rate? 

2. What happens to the training if we increase the number of images but NOT the number of classes? Supply a reasonable explanation and show an informative graph(s)

3. What happens to the training time if we increase the images' size? What would happen if the "device" variable was set differently?

4. How would you support different dataset images size (different input for "FakeData"'s constructor's "image_size" parameter), without modifying your model?

5. Consider adding a validation set. How would you expect its loss to behave? How would you cause it to behave differently?

6. How would you measure your model robustness to noise?

7. Write your own question.


"""