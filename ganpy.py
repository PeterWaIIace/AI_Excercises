import torch
from torch import nn

class GAN:

    def __init__(self,discriminator,generator):
        self.disc          = discriminator
        self.gen           = generator
        self.epoch_counter = 0
        pass

    def checkDevice(self):
        self.device = ""
        if torch.cuda.is_available():
            print("Using GPU with CUDA")
            self.device = torch.device("cuda")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

    def setParams(self,lr = 0.0001 ,epochs = 50,loss_functions = nn.BCELoss()):
        self.lr = lr
        self.num_epochs = epochs
        self.loss_function = loss_functions

    def prepareTrainSet(self,batch_size,train_set):
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

    def prepareOptimizer(self):
        self.optimizer_disc = torch.optim.Adam(self.disc.parameters(), lr=self.lr)
        self.optimizer_gen  = torch.optim.Adam(self.gen.parameters(), lr=self.lr)

    def stepDiscriminator(self,real_samples):
        # Data for training the discriminator
        real_samples            = real_samples.to(device=self.device)
        real_samples_labels     = torch.ones((self.batch_size, 1)).to(device=self.device)
        latent_space_samples    = torch.randn((self.batch_size, 2)).to(device=self.device)
        
        generated_samples        = self.gen(latent_space_samples)
        generated_samples_labels = torch.zeros((self.batch_size, 1)).to(device=self.device)
    
        all_samples         = torch.cat((real_samples, generated_samples))
        all_samples_labels  = torch.cat((real_samples_labels, generated_samples_labels))

        # Training the discriminator
        self.disc.zero_grad()
        output_discriminator = self.disc(all_samples)

        loss_discriminator = self.loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        
        self.optimizer_disc.step()

        return loss_discriminator

    def stepGenerator(self):
        # Data for training the generator
        real_samples_labels  = torch.ones((self.batch_size, 1)).to(device=self.device)
        latent_space_samples = torch.randn((self.batch_size, 2)).to(device=self.device)

        # Training the generator
        self.gen.zero_grad()
        generated_samples = self.gen(latent_space_samples)

        output_discriminator_generated = self.disc(generated_samples)
        loss_generator = self.loss_function( output_discriminator_generated, real_samples_labels )
        
        loss_generator.backward()
        self.optimizer_gen.step()

        return loss_generator

    def generate(self):
        latent_space_samples = torch.randn((100, 2)).to(device=self.device)
        generated_samples = self.gen(latent_space_samples)

        return generated_samples.detach()

    def train(self):

        self.checkDevice()
        for epoch in range(self.num_epochs):
            for n, (real_samples, mnist_labels) in enumerate(self.train_loader):
                loss_discriminator = self.stepDiscriminator(real_samples)
                loss_generator     = self.stepGenerator()

                # Show loss
                if n == self.batch_size - 1:
                    print(f"Epoch: {self.epoch_counter} Loss D.: {loss_discriminator}")
                    print(f"Epoch: {self.epoch_counter} Loss G.: {loss_generator}")
            self.epoch_counter += 1        
                