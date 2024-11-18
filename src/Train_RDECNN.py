import torch
import time
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from RDECNN import RDENet
from Polluted_Images_Generation import CRRNWEP

class FashionMNIST_RDECNNClassifier:
    def __init__(self, batch_size=64, lr=1e-3, epochs=10, device=None, save_path='../models/RDECNN'):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path

        # Data preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Anti-normalize for visualization
        self.inv_normalize = transforms.Normalize(
            mean=(-0.5 / 0.5,), std=(1 / 0.5,)
        )

        # Define the transform with normalization and pollution
        self.polluting_transform = transforms.Compose([
            CRRNWEP(range1=(-30, -10), range2=(10, 30), size=(28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Load data
        self.train_loader, self.test_loader, self.polluted_train_loader, self.polluted_test_loader, self.mixed_train_loader = self.load_data()

        # Initialize model
        self.model = self.initialize_model()

        # Loss function, optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        print('Using device:', self.device)

    def set_path(self, path):
        self.save_path = path

    def load_data(self):
        train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        polluted_train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=self.polluting_transform)
        polluted_test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=self.polluting_transform)

        mixed_train_dataset = ConcatDataset([train_dataset, polluted_train_dataset])


        polluted_train_loader = DataLoader(polluted_train_dataset, batch_size=self.batch_size, shuffle=True)
        polluted_test_loader = DataLoader(polluted_test_dataset, batch_size=self.batch_size, shuffle=False)
        mixed_train_loader = DataLoader(mixed_train_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, polluted_train_loader, polluted_test_loader, mixed_train_loader

    def initialize_model(self):
        self.params = dict(
            in_size=(1, 28, 28),  # Input shape of FashionMNIST
            out_classes=10,  # FashionMNIST has 10 classes
            channels=[16, 32, 64],  # Channels in each convolutional layer
            pool_every=2,  # Apply pooling after every 2 layers
            hidden_dims=[128, 64],  # Fully connected layers
            conv_params=dict(kernel_size=3, stride=1, padding=1),
            activation_type='relu',
            activation_params=dict(),
            pooling_type='max',
            pooling_params=dict(kernel_size=2, stride=2),
            batchnorm=True, 
            dropout=0.1,
            bottleneck=False
        )
        model = RDENet(**self.params).to(self.device)
        return model

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            save_checkpoints = f"{self.save_path}-epoch-{epoch}.pt"
            running_loss = 0.0
            correct, total = 0, 0

            with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    # Compute loss and accuracy
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            epoch_duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
                  f"Duration: {epoch_duration:.2f}s")
            torch.save(self.model.state_dict(), save_checkpoints)

    def test(self):
        self.model.eval()
        correct, total = 0, 0
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    def polluted_train(self):
        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            save_checkpoints = f"{self.save_path}-epoch-{epoch}.pt"
            running_loss = 0.0
            correct, total = 0, 0

            with tqdm(self.polluted_train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    # Compute loss and accuracy
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            epoch_duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
                  f"Duration: {epoch_duration:.2f}s")
            torch.save(self.model.state_dict(), save_checkpoints)

    def mixed_train(self):
        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            save_checkpoints = f"{self.save_path}-epoch-{epoch}.pt"
            running_loss = 0.0
            correct, total = 0, 0

            with tqdm(self.mixed_train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    # Compute loss and accuracy
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            epoch_duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
                  f"Duration: {epoch_duration:.2f}s")
            torch.save(self.model.state_dict(), save_checkpoints)    


    def polluted_test(self):
        self.model.eval()
        correct, total = 0, 0
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.polluted_test_loader, desc="Testing", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                

        avg_loss = running_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")    


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print("Loaded model from", path)



if __name__ == "__main__":
    classifier = FashionMNIST_RDECNNClassifier(batch_size=64, lr=1e-3, epochs=20)
    print(classifier.model)
    # classifier.train()
    classifier.mixed_train()
    classifier.test()
    classifier.polluted_test()
