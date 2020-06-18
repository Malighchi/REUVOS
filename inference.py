import config
import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
from model import VOSModel
from dataloader import ValidationDataset

def inference(model_path = './SavedModels/folder/model.pth'):
	criterion = nn.BCELoss(reduction='mean')
	model = VOSModel()
	load_model = torch.load(model_path)
	model.load_state_dict(load_model['state_dict'])
	if config.use_cuda:
		model.cuda()
	model.eval()
	print("model loaded...")

	with torch.no_grad():
		valid_dataset = ValidationDataset()
		print('dataset loaded...')
		for video in range(170, len(valid_dataset.valid_frames)):
			segs_concat = np.array([])
			for object in range(len(valid_dataset.valid_frames[video])):
				frame = 0
				y_pred_concat = np.array([])
				while frame < len(valid_dataset.valid_frames[video][object]):
					# Batch-size of model during inference would be 1, because we compute segmentations for 1 video at-a-time
					if(frame == 0):
						#initial_frame_path = valid_dataset.valid_annotations[video][object][0].replace('Annotations', 'JPEGImages')
						#initial_frame_path = initial_frame_path.replace('png', 'jpg')
						#frame = valid_dataset.valid_frames[video][object].index(initial_frame_path)
						video_inputs_batch, video_annotations_batch, video_annotations_indeces_batch = valid_dataset.Datalaoder(video_index=video, object_index=object, initial_frame=frame)
						last_frame = video_annotations_batch[:, :, 0]
					else:
						last_frame = y_pred[:, :, -1]
						video_inputs_batch, video_annotations_batch, video_annotations_indeces_batch = valid_dataset.Datalaoder(video_index=video, object_index=object, initial_frame=frame)
					#print(video_inputs_batch.shape)
					#print(video_annotations_batch.shape)
					#print(video_annotations_indeces_batch.shape)
					#last_array = torch.round(last_frame).cpu().numpy().astype(np.uint8)
					y_pred, _ = model(video_inputs_batch, video_inputs_batch[:, :, 0], last_frame)
					loss = criterion(y_pred[:, :, video_annotations_indeces_batch[0][0], :, :], video_annotations_batch[:, :, video_annotations_indeces_batch[0][0], :, :])
					print(loss.item())
					frame += 32
					if y_pred_concat.size == 0:
						y_pred_concat = torch.round(y_pred).cpu().numpy().astype(np.uint8)
					else:
						y_pred_concat = np.concatenate((y_pred_concat, torch.round(y_pred).cpu().numpy().astype(np.uint8)), axis=2)
				if segs_concat.size == 0:
					segs_concat = y_pred_concat
				else:
					segs_concat = np.concatenate((segs_concat, y_pred_concat), axis=1)
					#print("finsihed object segmentation...")
					#annotation = (y_pred[0][0][0].cpu().numpy() * 255).astype(np.uint8)
					#c = Image.fromarray(annotation, mode='P')
					#c.putpalette(Image.ADAPTIVE)
					#c.save(str(frame) + '.png', "PNG", mode='P')
					#print("segmentation saved...")
			segs_concat = (segs_concat.squeeze(0))
			mask_for_frames = np.argmax(segs_concat, axis=0)
			print("finsihed object segmentation...")
			video_file = valid_dataset.valid_frames[video][0][0].split('/')
			try:
				dir = 'Annotations/%s/' % video_file[-2]
				os.mkdir(dir)
			except:
				pass
			palette = Image.open(valid_dataset.valid_annotations[video][0][0])
			for images in range(len(valid_dataset.valid_frames[video][0])):
				img_file = valid_dataset.valid_frames[video][0][images].split('/')
				c = Image.fromarray(mask_for_frames[images], mode='P').resize(size=palette.size)
				c.putpalette(palette.getpalette())
				img_path = dir + img_file[-1]
				img_path = img_path.replace('jpg', 'png')
				c.save(img_path, "PNG", mode='P')
			print("segmentation saved...")
	
if __name__ == '__main__':
    inference(config.model_path)