import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import models, transforms, datasets
from sklearn.metrics import auc, roc_curve
from datetime import datetime
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

class PAN_AI_X:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(weights_path).to(self.device)
        self.model.eval()
        self.transformval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transformtrain = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, weights_path):
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        model.load_state_dict(torch.load(weights_path))
        return model
        
    def _load_model_train(self):
        model = models.resnet50(weights='IMAGENET1K_V2')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        return model

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transformval(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output).item()
        
        return prediction

    def train(self, train_dataset, val_dataset, test_dataset):
        self.model = self._load_model_train().to(self.device)
        
        train_set = datasets.ImageFolder(root=train_dataset, transform=self.transformtrain)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_set = datasets.ImageFolder(root=val_dataset, transform=self.transformval)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        test_set = datasets.ImageFolder(root=test_dataset, transform=self.transformval)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        writer = SummaryWriter('runs/ToothPredictor_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        epoch_number = 0
        num_epochs = 50
        best_auc = 0.0

        print('TRAINING:')
        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch_number + 1))
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                labels = labels.float()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}")

            self.model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            predictions = []
            true_labels = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    labels = labels.float()
                    val_running_loss += criterion(outputs.squeeze(), labels).item()
                    probabilities = torch.sigmoid(outputs).squeeze()
                    predicted = (probabilities >= 0.5).float()
                    predictions.extend(probabilities.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss = val_running_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            fpr, tpr, _ = roc_curve(true_labels, predictions)
            val_auc = auc(fpr, tpr)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%, Validation AUC: {val_auc}")

            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': running_loss / len(train_loader), 'Validation': val_loss},
                               epoch_number + 1)
            writer.add_scalar('Validation Acc', val_accuracy, epoch_number + 1)
            writer.add_scalar('Validation AUC', val_auc, epoch_number + 1)
            writer.flush()
            epoch_number += 1

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), 'ToothPredictor_BestAUC.pth')
        
        # Testing
        print('TESTING:')
        self.model.eval()
        test_predictions = []
        test_true_labels = []
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs).squeeze()

                # Threshold probabilities to get binary predictions
                predicted = (probabilities >= 0.5).float()

                # Update accuracy count
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Extend predictions and true labels lists
                test_predictions.extend(probabilities.cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())

        # Calculate test accuracy
        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy}%")

        # Calculate test AUC
        test_fpr, test_tpr, _ = roc_curve(test_true_labels, test_predictions)
        test_auc = auc(test_fpr, test_tpr)
        print(f"Test AUC: {test_auc}")
        
        # Save the final trained model
        torch.save(self.model.state_dict(), 'ToothPredictor_final.pth')


# Example usage:
# weights_path = '/.../model.pth'
# image_path = '/.../image.png'
# train_dataset_path = '/.../train_dir'
# val_dataset_path = '/.../val_dir'
# test_dataset_path = '/.../test_dir'
#
# Create an instance of ToothPredictor
# tooth_predictor = PAN_AI_X(weights_path)
#
# Use predict method for inference
# prediction = tooth_predictor.predict(image_path)
# print(prediction)
#
# Use train method for training
# tooth_predictor.train(train_dataset_path, val_dataset_path, test_dataset_path)
