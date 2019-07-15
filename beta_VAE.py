import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(beta, total_epoch):
    # Dataloading
    batch_size = 64
    img_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root='./MNIST_data', train=True, transform=img_transform, download=True)
    test_dataset = MNIST(root='./MNIST_data', train=False, transform=img_transform, download=True)

    train_data = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=10000, num_workers=4, pin_memory=True, shuffle=True)

    input_size = 28 * 28

    # Create the Network
    class beta_VAE(nn.Module):
        def __init__(self, latent_dim, obs_latent=False):
            super(beta_VAE, self).__init__()
            self.latent_dim = latent_dim
            self.obs_latent = obs_latent
            self.mu = None
            self.sigma = None
            self.latent_var = None
            self.training = True

            # Encoder
            self.encode = nn.Sequential(
                nn.Linear(input_size, 512), nn.ELU(),
                nn.Linear(512, 256), nn.Tanh(),
                nn.Linear(256, 2*latent_dim)
            )

            # Decoder
            self.decode = nn.Sequential(
                nn.Linear(latent_dim + 10, 256), nn.Tanh(),
                nn.Linear(256, 512), nn.ELU(),
                nn.Linear(512, input_size), nn.Sigmoid()
            )

        def forward(self, x):
            activ_fun = nn.Softplus()
            phi = self.encode(x[0])
            self.mu = phi[:,:self.latent_dim]
            self.sigma = 1e-6 + activ_fun(phi[:,self.latent_dim:])
            self.latent_var = self.mu + self.sigma * torch.randn(self.latent_dim).to(device)

            x_label_one_hot = torch.zeros((x[0].shape[0],10))
            index = torch.tensor([i for i in range(x[0].shape[0])])
            x_label_one_hot[index, x[1]] = 1
            output = self.decode(torch.cat((self.latent_var, x_label_one_hot), dim=1))
            if self.obs_latent:
                return output, self.latent_var, self.mu, self.sigma
            return output


    my_VAE = beta_VAE(latent_dim=2, obs_latent=True).to(device)
    optimizer = torch.optim.Adam(my_VAE.parameters())

    # Train beta_VAE
    num_epoch = total_epoch  # 1 epoch = uses all data points once
    beta = beta

    for epoch in range(num_epoch):
        for data in train_data:
            input_train, label_train = data
            input_train = input_train.view(input_train.size(0), -1)
            input_train = input_train.to(device)

            # forward
            output_train = my_VAE([input_train, label_train])[0]

            Recon_score = (torch.sum(input_train * torch.log(output_train) + (1 - input_train) * torch.log(1 - output_train))).to(device)
            KL_div = (0.5 * torch.sum(my_VAE.mu**2 + my_VAE.sigma**2 - torch.log(1e-8 + my_VAE.sigma**2) - 1)).to(device)
            cost_train = (beta * KL_div - Recon_score) / batch_size # For reducing a large loss value
            cost_train.to(device)

            # backward
            optimizer.zero_grad()   # initialize gradients to be zero
            cost_train.backward()   # back-propagation
            optimizer.step()        # update weights using optimizer

        print("epoch [{}/{}], training loss VAE: {:.4f}".format(epoch+1, num_epoch, cost_train.data))

    # Test VAE
    for data in test_data:
        input_test, label_test = data
        input_test = input_test.view(input_test.size(0), -1)

        output_test, latent_test, latent_mu, latent_sigma = my_VAE([input_test, label_test])
        Recon_score = torch.sum(input_test * torch.log(output_test) + (1 - input_test) * torch.log(1-output_test))
        KL_div = 0.5 * torch.sum(my_VAE.mu ** 2 + my_VAE.sigma ** 2 - torch.log(1e-8 + my_VAE.sigma ** 2) - 1)
        cost_test = (beta * KL_div - Recon_score) / 10000

        print(Recon_score.data, KL_div.data)
        print("test loss beta_VAE: {:.4f}".format(cost_test.data))

    latent_array = latent_test.detach().numpy()
    fig0 = plt.figure(figsize=(7,7))
    plt.title('2D latent space of beta_VAE with beta={}'.format(beta))
    plt.scatter(latent_array[:,0], latent_array[:,1], c=label_test, cmap='nipy_spectral')
    plt.clim(0,9)
    plt.colorbar()
    plt.show()
    fig0.savefig('2D latent space of beta_VAE with beta='+str(beta)+'.jpg')

    fig1 = plt.figure(figsize=(5,5))
    plt.title('Original image with beta={}'.format(beta))
    idx = random.choice([i for i in range(10000)])
    original_img = input_test[idx]
    original_img = original_img.detach().numpy().reshape((28,28))
    plt.imshow(original_img, cmap='gray')
    plt.show()
    fig1.savefig('Original image with beta='+str(beta)+'.jpg')

    fig2 = plt.figure(figsize=(5,5))
    plt.title('Reconstructed image with beta={}'.format(beta))
    recon_img = output_test[idx]
    recon_img = recon_img.detach().numpy().reshape((28,28))
    plt.imshow(recon_img, cmap='gray')
    plt.show()
    fig2.savefig('Reconstructed image with beta='+str(beta)+'.jpg')

    if idx == 0:
        new_idx = 5
    new_idx = (idx * 3) % 10

    fig3 = plt.figure(figsize=(5,5))
    plt.title('Reconstructed image with different discrete latent with beta={}'.format(beta))
    new_one_hot_label = torch.zeros(10)
    new_one_hot_label[new_idx] = 1
    recon_diff_disc_img = my_VAE.decode(torch.cat((latent_test[new_idx], new_one_hot_label)))
    recon_diff_disc_img = recon_diff_disc_img.detach().numpy().reshape((28,28))
    plt.imshow(recon_diff_disc_img, cmap='gray')
    plt.show()
    fig3.savefig('Reconstructed image with different discrete latent with beta='+str(beta)+'.jpg')

    output_array_first = np.zeros((28 * 10, 28 * 10))
    output_array_second = np.zeros((28 * 10 , 28 * 10))
    for repeat in range(10):
        for target_number in range(10):
            one_hot_label = torch.zeros(10)
            one_hot_label[target_number] = 1

            epsilon = torch.tensor([-2.5 + 0.5*repeat, 0])    # This will perturb only the 1st coordinate of the latent_test
            recon_diff_discrete_first_img = my_VAE.decode(torch.cat((latent_test[idx]+epsilon, one_hot_label)))
            recon_diff_discrete_first_img = recon_diff_discrete_first_img.detach().numpy().reshape((28,28))

            delta = torch.tensor([0, -2.5 + 0.5*repeat])      # This will perturb only the 2nd coordinate of the latent_test
            recon_diff_discrete_second_img = my_VAE.decode(torch.cat((latent_test[idx]+delta, one_hot_label)))
            recon_diff_discrete_second_img = recon_diff_discrete_second_img.detach().numpy().reshape((28,28))

            output_array_first[28*repeat:28*(repeat+1), 28*target_number:28*(target_number+1)] = recon_diff_discrete_first_img
            output_array_second[28*repeat:28*(repeat+1), 28*target_number:28*(target_number+1)] = recon_diff_discrete_second_img

    fig4 = plt.figure(figsize=(5,5))
    plt.title('1st coordinate perturbed with beta={}'.format(beta))
    plt.imshow(output_array_first, cmap='gray')
    plt.show()
    fig4.savefig('1st coordinate perturbed with beta='+str(beta)+'.jpg')

    fig5 = plt.figure(figsize=(5,5))
    plt.title('2nd coordinate perturbed with beta={}'.format(beta))
    plt.imshow(output_array_second, cmap='gray')
    plt.show()
    fig5.savefig('2nd coordinate perturbed with beta='+str(beta)+'.jpg')



for i in range(1):
    main(beta=1.6, total_epoch=1)
