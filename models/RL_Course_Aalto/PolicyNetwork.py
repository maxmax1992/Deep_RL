import torch
import torch.nn as nn
import numpy as np


class ConvNetPG(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    def forward(self, x):
        # x = x.view(-1, x.shape[0])
        x = self.classifier(x)
        return x

    def train(self, X, y, n_steps, batch_size=32, total_reward=1):
        N = X.shape[0]
        batch_size = min(batch_size, N)

        for t in range(n_steps):
            batch_indices = np.random.randint(0, N, batch_size)
            # print(X_.size())

            y_pred = self.forward(X[batch_indices])


            # Compute and print loss.
            loss = self.loss_fn(y_pred, y[batch_indices])
            #             print(loss.item())

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            print('lossloss', loss)
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step(score=total_reward)