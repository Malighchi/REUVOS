#from matplotlib import pyplot
import os
import config
import json
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, root='./data/train/'):
        self.train_frames = []
        self.train_annotations = []

        with open(os.path.join(root, 'meta.json')) as f_train:
            train_data = json.load(f_train)

        for video_folders in train_data['videos']:
            video_frames = []
            video_annotations = []
            for objects in train_data['videos'][video_folders]['objects']:
                object_frames = []
                object_annotations = []

                video_all_frames = os.listdir(os.path.join(root, 'JPEGImages/', video_folders + '/'))
                for frames in video_all_frames:
                    frame = os.path.join(root, 'JPEGImages/', video_folders + '/', frames)
                    object_frames.append(frame)
                for frames in train_data['videos'][video_folders]['objects'][objects]['frames']:
                    #frame = os.path.join(root, 'train/JPEGImages/', video_folders + '/', frames + '.jpg')
                    annotation = os.path.join(root, 'Annotations/', video_folders + '/', frames + '.png')
                    #object_frames.append(frame)
                    object_annotations.append(annotation)

                video_frames.append(object_frames)
                video_annotations.append((object_annotations))
            self.train_frames.append(video_frames)
            self.train_annotations.append(video_annotations)

    def __len__(self):
        return len(self.train_frames)

    def __getitem__(self, index):
        img_size_x = 224
        img_size_y = 224
        train_sample_frames = []#torch.zeros(32, 224, 224, 3)
        train_sample_annotations = []  # torch.zeros(1, 224, 224)
        train_sample_annotations.append([])
        train_sample_annotations_indeces = []#torch.zeros(224, 224, 3)
        train_sample_annotations_indeces.append([])

        object = random.randint(0, len(self.train_frames[index])-1)
        rand_ann = len(self.train_annotations[index][object]) - 8
        if rand_ann < 0:
            rand_ann = 0
        initial_annotation = random.randint(0,rand_ann)
        initial_frame_path = self.train_annotations[index][object][initial_annotation].replace('Annotations', 'JPEGImages')
        initial_frame_path = initial_frame_path.replace('png', 'jpg')
        initial_frame = self.train_frames[index][object].index(initial_frame_path)

        for frames in range(initial_frame, (initial_frame+32)):
            try:
                frame = np.array(Image.open(self.train_frames[index][object][frames]))
                if(frames-initial_frame == 0):
                    vid_height, vid_width, _ = frame.shape
                if(vid_width >vid_height):
                    new_height_min = 0
                    new_height_max = vid_height
                    new_width_min = int(((vid_width/2) - (vid_height/2)))
                    new_width_max = int(((vid_width/2) + (vid_height/2)))
                elif(vid_width < vid_height):
                    new_height_min = int(((vid_height/2) - (vid_width/2)))
                    new_height_max =  int(((vid_height/2) + (vid_width/2)))
                    new_width_min = 0
                    new_width_max = vid_width
                else:
                    new_height_min = 0
                    new_height_max = vid_height
                    new_width_min = 0
                    new_width_max = vid_width

                annotation_path = self.train_frames[index][object][frames].replace('JPEGImages', 'Annotations')
                annotation_path = annotation_path.replace('jpg', 'png')
                if(os.path.exists(annotation_path)):
                    annotation = np.array(Image.open(annotation_path))
                    annotation[annotation != (object + 1)] = 0
                    train_sample_annotations_indeces[0].append(frames-initial_frame)
                    #print(frames)
                else:
                    annotation = np.zeros((vid_height, vid_width))

                #annotation = annotation[new_height_min:new_height_max, new_width_min:new_width_max]
                #frame = frame[new_height_min:new_height_max, new_width_min:new_width_max]

                frame = np.array(Image.fromarray(frame).resize(size=(img_size_x, img_size_y)))
                annotation = np.array(Image.fromarray(annotation).resize(size=(img_size_x, img_size_y)))

                train_sample_frames.append(frame)
                train_sample_annotations[0].append(annotation)
            except:
                train_sample_frames.append(train_sample_frames[-1])
                train_sample_annotations[0].append(train_sample_annotations[0][-1])

        while(len(train_sample_annotations_indeces[0]) < 7):
            train_sample_annotations_indeces[0].append(train_sample_annotations_indeces[0][-1])
        while (len(train_sample_annotations_indeces[0]) > 7):
            train_sample_annotations_indeces[0].pop()

        train_sample_frames = np.stack(train_sample_frames, 0)
        train_sample_frames = np.transpose(train_sample_frames, (3, 0, 1, 2))
        #print(train_sample_frames.shape)
        train_sample_annotations = np.stack(train_sample_annotations, 0)
        #print(train_sample_annotations.shape)
        train_sample_annotations_indeces = np.stack(train_sample_annotations_indeces, 0)
        #print(train_sample_annotations_indeces.shape)

        train_sample_frames = torch.from_numpy(train_sample_frames).type(torch.float)
        train_sample_annotations = torch.from_numpy(train_sample_annotations).type(torch.float)
        train_sample_annotations_indeces = torch.from_numpy(train_sample_annotations_indeces).type(torch.float)

        return train_sample_frames, train_sample_annotations, train_sample_annotations_indeces

#seperate changes
class ValidationDataset(Dataset):
    def __init__(self, root='./data/valid/'):
        self.valid_frames = []
        self.valid_annotations = []

        with open(os.path.join(root, 'meta.json')) as f_valid:
            valid_data = json.load(f_valid)

        for video_folders in valid_data['videos']:
            video_frames = []
            video_annotations = []
            for objects in valid_data['videos'][video_folders]['objects']:
                object_frames = []
                object_annotations = []

                video_all_frames = os.listdir(os.path.join(root, 'JPEGImages/', video_folders + '/'))
                for frames in video_all_frames:
                    frame = os.path.join(root, 'JPEGImages/', video_folders + '/', frames)
                    object_frames.append(frame)
                for frames in valid_data['videos'][video_folders]['objects'][objects]['frames']:
                    #frame = os.path.join(root, 'valid/JPEGImages/', video_folders + '/', frames + '.jpg')
                    annotation = os.path.join(root, 'Annotations/', video_folders + '/', frames + '.png')
                    #object_frames.append(frame)
                    object_annotations.append(annotation)

                video_frames.append(object_frames)
                video_annotations.append((object_annotations))
            self.valid_frames.append(video_frames)
            self.valid_annotations.append(video_annotations)

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, index):
        img_size_x = 224
        img_size_y = 224
        valid_sample_frames = []    #torch.zeros(32, 224, 224, 3)
        valid_sample_annotations = []   #torch.zeros(1, 224, 224)
        valid_sample_annotations.append([])
        valid_sample_start_frame_indeces = []   #torch.zeros(224, 224, 3)
        valid_sample_start_frame_indeces.append([])

        for frames in range(len(self.valid_frames[index][0])):
            try:
                frame = np.array(Image.open(self.valid_frames[index][0][frames]))
                frame = np.array(Image.fromarray(frame).resize(size=(img_size_x, img_size_y)))

                valid_sample_frames.append(frame)
            except:
                valid_sample_frames.append(valid_sample_frames[-1])
        for objects in range(len(self.valid_frames[index])):
            annotation_path = self.valid_annotations[index][objects][0]
            annotation = np.array(Image.open(annotation_path))
            annotation[annotation != (objects + 1)] = 0
            annotation = np.array(Image.fromarray(annotation).resize(size=(img_size_x, img_size_y)))

            initial_frame_path = annotation_path.replace('Annotations', 'JPEGImages')
            initial_frame_path = initial_frame_path.replace('png', 'jpg')
            initial_frame = self.valid_frames[index][objects].index(initial_frame_path)

            valid_sample_start_frame_indeces[0].append(initial_frame)
            valid_sample_annotations[0].append(annotation)

        valid_sample_frames = np.stack(valid_sample_frames, 0)
        valid_sample_frames = np.transpose(valid_sample_frames, (3, 0, 1, 2))
        #print(valid_sample_frames.shape)
        valid_sample_annotations = np.stack(valid_sample_annotations, 0)
        #print(valid_sample_annotations.shape)
        valid_sample_annotations_indeces = np.stack(valid_sample_start_frame_indeces, 0)
        #print(valid_sample_annotations_indeces.shape)

        valid_sample_frames = torch.from_numpy(valid_sample_frames).type(torch.float)
        valid_sample_annotations = torch.from_numpy(valid_sample_annotations).type(torch.float)
        valid_sample_annotations_indeces = torch.from_numpy(valid_sample_annotations_indeces).type(torch.float)
        return valid_sample_frames, valid_sample_annotations, valid_sample_annotations_indeces

if __name__ == '__main__':
    valid = ValidationDataset()
    for i in range(len(valid)):
        x, y, z = valid.__getitem__(i)