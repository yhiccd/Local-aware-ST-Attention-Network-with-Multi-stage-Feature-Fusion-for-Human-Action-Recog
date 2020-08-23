
# Network Architecture

import os
import numpy as np
import cv2 as cv
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from pretrainedmodels.models import bninception
from framepath import Path
import optical_flow 
import STCB2
import STCB3
import time
import copy
import re
import csv

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class STPN(torch.nn.Module):

	# Inception models expects tensors with a size of N x 3 x 300 x 224

	def __init__(self, num_classes = 2):
		super(STPN, self).__init__()

		# CNNs
		self.cnn1 = bninception(pretrained = 'imagenet')
		self.cnn2 = bninception(pretrained = 'imagenet')
		self.cnn3 = bninception(pretrained = 'imagenet')
		self.cnn4 = bninception(pretrained = 'imagenet')
        
		self.cnn1 = self.cnn1.cuda()
		self.cnn2 = self.cnn2.cuda()
		self.cnn3 = self.cnn3.cuda()
		self.cnn4 = self.cnn4.cuda()


		# Spatial Stream
		self.avgPool1 = torch.nn.AvgPool2d((7, 7))

		# Temporal Stream
		self.avgPool2 = torch.nn.AvgPool2d((7, 7))

		# Attention stream
		# STCB layers
		self.stcb1 = STCB3.CompactBilinearPooling(input_dim1 = 1024, input_dim2 = 1024, input_dim3 = 1024, output_dim = 1024)
		self.stcb1.cuda()
		self.stcb1.train()

		self.stcb2 = STCB2.CompactBilinearPooling(input_dim1 = 1024, input_dim2 = 1024, output_dim = 2048)
		self.stcb2.cuda()
		self.stcb2.train()
		
		# Convolutional layers
		self.conv1 = torch.nn.Conv2d(2048, 64, (1, 1)).cuda()
		self.conv2 = torch.nn.Conv2d(64, 1, (1, 1)).cuda()
		self.sm = torch.nn.Softmax2d().cuda()

		# Weighted Pooling Layer
		self.wtPool = torch.nn.AvgPool2d((7, 7))

		# Intersection of streams
		self.stcb3 = STCB3.CompactBilinearPooling(input_dim1 = 1024, input_dim2 = 1024, input_dim3 = 1024, output_dim = 4096)
		self.stcb3.cuda()
		self.stcb3.train()

		self.fc = torch.nn.Linear(4096, num_classes)
		self.lrelu = torch.nn.LeakyReLU()
		self.dropout = torch.nn.Dropout(0.5)
		self.tanh = torch.nn.Tanh()      
        
	def forward(self, rgb1, of1, of2, of3):
		# Spatial Stream
		rgb = self.cnn1.features(rgb1)
		rgb = torch.squeeze(rgb)
		spat = self.avgPool1(rgb)	# Average pooling for the spatial stream
        
		# Temporal Stream
		of1 = self.cnn2.features(of1)    
		of2 = self.cnn3.features(of2)
		of3 = self.cnn4.features(of3)        

		of = self.stcb1(of1, of2, of3)
		of = torch.mean(of, 0).cuda()        
		temp = self.avgPool2(of)	# Average pooling for the temporal stream

		# Attention Stream
		att1 = self.stcb2(rgb, of)
		att1.unsqueeze_(0)       
		att1 = self.conv1(att1)		# 1st convolution layer
		att1 = self.conv2(att1)		# 2nd convolution layer
		att1 = self.sm(att1)		# Softmax layer
		att = self.wtPool(att1 * rgb)	# Weighted pooling
        
		spat.unsqueeze_(0) 
		temp.unsqueeze_(0) 
		spatemp = self.stcb3(spat, temp, att)

		spatemp = torch.squeeze(spatemp)        
		spatemp_lrelu = self.lrelu(spatemp)
		spatemp_tanh = self.tanh(spatemp_lrelu)
		spatemp_dropout = self.dropout(spatemp_tanh)
		res = self.fc(spatemp_dropout)

		return res


def getpaths(dataset='ucf101', split='train'):
    root_dir = '/notebooks/storage/dataset/ucf3_post_split'
    folder = os.path.join(root_dir, split)
    data_paths = {}
    for label in sorted(os.listdir(folder)):
        for foldername in sorted(os.listdir(os.path.join(folder, label))):
            file_path = os.path.join(folder, label, foldername)
            data_paths.update({file_path:label})

    return data_paths

def printEpoch(epoch, num_epoch, start):
    string = "Epoch {}/{} : {} - {:.0f}h {:.0f}m\n{}\n".format(epoch, num_epoch, time.asctime(),
                                                             (time.time() - start)//3600, ((time.time() - start)//60)%60, '-' * 10)
    return string

def printStats(phase, loss, acc, best_acc):
    string = "{} Loss: {:.4f} Epoch Acc: {:.4f} Best Acc: {:.4f}".format(phase, loss, acc, best_acc)
    return string

def train_model(model, num_epochs=25, learning_rate = 0.01):
    since = time.time()
    PATH = '/notebooks/storage/checkpoint2_1234.pt'
    logPath = '/notebooks/storage/log2_1234.txt'
    criterion, optimizer, _ = createLossAndOptimizer(model, learning_rate)
    
    # Loading checkpoint if it exists
    if os.path.exists(PATH):
        print("Checkpoint loaded")
        checkpoint = torch.load(PATH)
        model.state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        epoch_loss = checkpoint['loss']
        best_acc = checkpoint['best_acc']
        best_model_wts = checkpoint['best_wts']
    else:
        best_acc = 0.0
        current_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
    
    data_paths_train = getpaths(dataset='ucf101', split ='train')
    data_paths_val = getpaths(dataset='ucf101', split ='test')
    labels = {}

    # Read the labels
    with open('/notebooks/storage/dataset/ucf3_labels.txt', 'r') as f:
        for line in f:
            splitLine = line.split()
            labels[splitLine[1]] = int(splitLine[0])
   
    for epoch in range(current_epoch, num_epochs):
    	# Adjust the learning rate based on the epoch
        if epoch < 100:
            learning_rate = 0.01
        elif epoch < 300: 
            learning_rate = 0.001
        elif epoch < 600:
            learning_rate = 0.0001
        else:
            learning_rate = 0.00001
            
        print(printEpoch(epoch, num_epochs - 1, since))
        if epoch == 0:
            # If first epoch, create text file and write to it
            doc = open(logPath, 'w')
            doc.write(printEpoch(epoch, num_epochs - 1, since))
        else:
            # If not the first epoch, open text file to append to it
            doc = open(logPath, 'a')
            doc.write("\n{}".format(printEpoch(epoch, num_epochs - 1, since)))

        dataset_sizes = {'train': 0, 'val': 0}
        # Each epoch has a training and validation phase        
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.            
            if phase == 'train':
                data_paths = data_paths_train
            else:
                data_paths = data_paths_val
                
            for file_paths, label in data_paths.items():
                dataset_sizes[phase] += 2
                label_value = labels[label]-1 
                label_1dtensor = torch.tensor([label_value]).cuda()
                label_tensor = torch.zeros([2]).cuda()
                label_tensor[label_value] = 1
                label_tensor = label_tensor.long()

                f1t, f2t, f3t, rgb = [], [], [], [1, 1]
                f1t.extend(([], []))
                f2t.extend(([], []))
                f3t.extend(([], []))

                if file_paths.split('/')[7] != '.ipynb_checkpoints':	# Ignore the .ipynb_checkpoints file if it exists
                    for frame in sorted(os.listdir(file_paths)):
                        img = cv.imread(os.path.join(file_paths, frame))
                        tensor = torch.from_numpy(img).cuda()
                        tensor = tensor.float()

                        num = int(re.search(r'\d+', frame).group())
                        cat = frame.split(str(num))[0]
                        # Group the corresponding frames together
                        if cat == "flow":
                            if num <= 10:
                                f1t[0].append(tensor)
                            elif num <= 20:
                                f2t[0].append(tensor)
                            elif num <= 30:
                                f3t[0].append(tensor)
                        elif cat == "flip":
                            if num <= 10:
                                f1t[1].append(tensor)
                            elif num <= 20:
                                f2t[1].append(tensor)
                            elif num <= 30:
                                f3t[1].append(tensor)
                        else:
                            if num == 1:
                                rgb[0] = tensor
                            else:
                                rgb[1] = tensor

                    flow1 = torch.stack([torch.stack(f1t[0]).cuda(), torch.stack(f1t[1]).cuda()]).cuda()
                    flow1 = flow1.permute(0, 1, 4, 3, 2)
                    flow2 = torch.stack([torch.stack(f2t[0]).cuda(), torch.stack(f2t[1]).cuda()]).cuda()
                    flow2 = flow2.permute(0, 1, 4, 3, 2)
                    flow3 = torch.stack([torch.stack(f3t[0]).cuda(), torch.stack(f3t[1]).cuda()]).cuda()
                    flow3 = flow3.permute(0, 1, 4, 3, 2)
                    rgb = torch.stack(rgb).cuda()
                    rgb = rgb.permute(0, 3, 2, 1)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    # Track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        for flip in range(len(rgb)):
                        	# Validate only on the un-augmented val set
                            if phase == 'val' and flip == 1:
                                break

                            outputs = model(rgb[flip].unsqueeze_(0), flow1[flip], flow2[flip], flow3[flip])
                            outputs.unsqueeze_(0)

                            #preds and labs return a single value which is the position of the max
                            label_tensor.unsqueeze_(0)                      
   
                            _, preds = torch.max(outputs, 1, keepdim = True)
                            _, labs = torch.max(label_tensor, 1, keepdim = True)

                            loss = criterion(outputs, label_1dtensor)

                            # Backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                            # Statistics
                            running_loss += loss.item()
                            running_corrects += torch.sum(preds == labs).cuda()

            if phase == 'val':
                dataset_sizes[phase] = dataset_sizes[phase]/2

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                torch.save(model, '/notebooks/storage/model_check_2_1234.pt')
                print("Model saved")
                
            print(printStats(phase, epoch_loss, epoch_acc, best_acc))
            # Update the log
            doc.write("{}\n".format(printStats(phase, epoch_loss, epoch_acc, best_acc)))
        doc.close()

        # Save the model every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_acc': best_acc,
                'best_wts': best_model_wts
            }, PATH)
            print("Checkpoint saved")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}hr {:.0f}m {:.0f}s'.format(
        time_elapsed//3600, (time_elapsed // 60) % 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open(logPath, 'a') as f:
        f.write("\nTraining complete in {:.0f}hr {:.0f}m {:.0f}s\nBest val acc: {:4f}\n".format(
            time_elapsed//3600, (time_elapsed // 60) % 60, time_elapsed % 60, best_acc))
        f.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
		

def createLossAndOptimizer(net, learning_rate = 0.01):
	# Loss function
	criterion = torch.nn.CrossEntropyLoss().cuda()

	# Optimizer
	optimizer = optim.Adam(net.parameters(), lr = learning_rate)

	scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

	return criterion, optimizer, scheduler


if __name__ == '__main__':
    PATH = '/notebooks/storage/model2_1234.pt'
    # Train model
    model_ft = train_model(STPN(num_classes = 2).cuda(), num_epochs=1000, learning_rate = 0.01)
    torch.save(model_ft, PATH)	# Save the model after training
