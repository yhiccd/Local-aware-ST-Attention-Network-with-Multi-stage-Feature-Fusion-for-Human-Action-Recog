# https://github.com/jfzhang95/pytorch-video-recognition/blob/master/dataloaders/dataset.py

import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
import re
import optical_flow

class Path():
	''' Retrieve the directory containing the data and the output directory for the processed data '''
	@staticmethod
	def db_dir(database):
		if database == 'ucf101':
			# Folder containing class labels
			root_dir = '/notebooks/storage/dataset/ucf'

			# Save preprocess data to output_dir
			output_dir = '/notebooks/storage/dataset/ucf4_post_split'

			return root_dir, output_dir

		elif database == 'hmdb51':
			root_dir = '/notebooks/storage/dataset/hmdb'
			output_dir = '/notebooks/storage/dataset/hmdb_post_split'

			return root_dir, output_dir
		else:
			print('Database {} not available.'.format(database))
			raise NotImplementedError

class VideoDataset(Dataset):
	''' Process and split the data '''
	def __init__(self, dataset = 'ucf101', split = 'train', preprocess = False):
		self.root_dir, self.output_dir = Path.db_dir(dataset)
		folder = os.path.join(self.output_dir, split)
		self.split = split

		if not self.check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
				' Please download it from the official website.')

		if (not self.check_preprocess()) or preprocess:
			print('Preprocessing of {} dataset. This will take long, but it will be done only once.'.format(dataset))
			self.preprocess()

		# Obtain all the filenames of files inside all the class folders
		# Goes through each class folder one at a time
		self.fnames, labels = [], []
		for label in sorted(os.listdir(folder)):
			for fname in os.listdir(os.path.join(folder, label)):
				self.fnames.append(os.path.join(folder, label, fname))
				labels.append(label)
		self.tnames, tlabels = [], []
		for label in sorted(os.listdir(os.path.join(self.output_dir, 'test'))):
			for tname in os.listdir(os.path.join(self.output_dir, 'test', label)):
				self.tnames.append(os.path.join(self.output_dir, 'test', label, tname))
				tlabels.append(label)
		assert len(labels) == len(self.fnames)
		print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

		assert len(tlabels) == len(self.tnames)
		print('Number of {} videos: {:d}'.format('test', len(self.tnames)))


		# Prepare a mapping between the label names (strings) and indices (ints)
		self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}

		# Convert the list of label names into an array of label indices
		self.label_array = np.array([self.label2index[label] for label in labels], dtype = int)

		path = '/notebooks/storage/dataset'
		if dataset == "ucf101":
			if not os.path.exists(os.path.join(path, 'ucf4_labels.txt')):
				with open(os.path.join(path, 'ucf4_labels.txt'), 'w') as f:
					for id, label in enumerate(sorted(self.label2index)):
						f.writelines(str(id + 1) + ' ' + label + '\n')
					f.close()
		elif dataset == "hmdb51":
			if not os.path.exists(os.path.join(path, 'hmdb_labels.txt')):
				with open(os.path.join(path, 'hmdb_labels.txt'), 'w') as f:
					for id, label in enumerate(sorted(self.label2index)):
						f.writelines(str(id + 1) + ' ' + label + '\n')
					f.close()

	def __len__(self):
		return len(self.fnames)

	def check_integrity(self):
		if not os.path.exists(self.root_dir):
			return False
		else:
			return True

	def check_preprocess(self):
		if not os.path.exists(self.output_dir):
			return False
		elif not os.path.exists(os.path.join(self.output_dir, 'train')):
			return False

		return True

	def gen_split(self, dataset = 'ucf101'): 
		# Specify the train and test list to read from
		trainlist, testlist = 'trainlist05.txt', 'testlist05.txt'

		train_list, test_list = [], []

		direct = "/notebooks/storage/dataset"

		# Read the train list
		if not os.path.exists(os.path.join(direct, trainlist)):
			print("Train list not available. Please include it in the dataset folder.")
		else:
			with open(os.path.join(os.path.join(direct, trainlist))) as f:
				train_list = f.readlines()

		# Read the test list
		if not os.path.exists(os.path.join(direct, testlist)):
			print("Test list not available. Please include it in the dataset folder.")
		else:
			with open(os.path.join(direct, testlist)) as f:
				test_list = f.readlines()

		# Grab only the video file names
		for files in range(len(train_list)):
			train_list[files] = train_list[files].split('.avi')[0] + ".avi"
		for files in range(len(test_list)):
			test_list[files] = test_list[files].split('.avi')[0] + ".avi"

		# Split train/test sets
		for folder in os.listdir(self.root_dir):
			file_path = os.path.join(self.root_dir, folder)		
			# video_files = list of videos in class folder
			video_files = [name for name in os.listdir(file_path)]

			train_dir = os.path.join(self.output_dir, 'train', folder)
			test_dir = os.path.join(self.output_dir, 'test', folder)

			# Creates folder for action in train/test split
			if not os.path.exists(train_dir):
				os.mkdir(train_dir)
			if not os.path.exists(test_dir):
				os.mkdir(test_dir)

			for video in train_list:
				action = video.split('/')[0]
				if action == folder:
					self.process_video(video, train_dir)

			for video in test_list:
				action = video.split('/')[0]
				if action == folder:
					self.process_video(video, test_dir)


	def preprocess(self):
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
			os.mkdir(os.path.join(self.output_dir, 'train'))
			os.mkdir(os.path.join(self.output_dir, 'test'))

		# Split train/val/test sets
		self.gen_split(dataset = 'ucf101')

		print('Preprocessing finished.')

	def process_video(self, video, save_dir):
		video_filename = video.split('/')[1].split('.')[0]
		
		# If the folder for the video doesn't exist, make it
		if not os.path.exists(os.path.join(save_dir, video_filename)):
			os.mkdir(os.path.join(save_dir, video_filename))

		save_path = os.path.join(save_dir, video_filename)
		# Obtain the optical flow for the current video
		if not os.path.exists(os.path.join(save_path, "flow30.jpg")):
			cap = cv.VideoCapture(os.path.join(self.root_dir, video))
			optical_flow.getInputs(cap, save_path)
			cap.release()

	def randomflip(self, buffer):
		'''
		Horizontally flip the give image and ground truth randomly with a probability of 0.5
		'''

		if np.random.random() < 0.5:
			for i, frame in enumerate(buffer):
				frame = cv.flip(buffer[i], flipCode = 1)
				buffer[i] = cv.flip(frame, flipCode = 1)

		return buffer

	def normalize(self, buffer):
		for i, frame in enumerate(buffer):
			frame -= np.array([[[90.0, 98.0, 102.0]]])
			buffer[i] = frame

		return buffer

if __name__ == "__main__":
	train_data = VideoDataset(dataset = 'ucf101', split = 'train', preprocess = True)
