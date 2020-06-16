import config
import torch
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
		for video in range(len(valid_dataset.valid_frames)):
			segs_concat = np.array([])
			for object in range(len(valid_dataset.valid_frames[video])):
				frame = 0
				y_pred_concat = np.array([])
				while frame < len(valid_dataset.valid_frames[video][object]):
					# Batch-size of model during inference would be 1, because we compute segmentations for 1 video at-a-time
					if(frame == 0):
						initial_frame_path = valid_dataset.valid_annotations[video][object][0].replace('Annotations', 'JPEGImages')
						initial_frame_path = initial_frame_path.replace('png', 'jpg')
						frame = valid_dataset.valid_frames[video][object].index(initial_frame_path)
						video_inputs_batch, video_annotations_batch, video_annotations_indeces_batch = valid_dataset.Datalaoder(video_index=video, object_index=object, initial_frame=frame)
						last_frame = video_annotations_batch[:, :, 0]
					else:
						last_frame = video_annotations_batch[:, :, -1]
						video_inputs_batch, video_annotations_batch, video_annotations_indeces_batch = valid_dataset.Datalaoder(video_index=video, object_index=object, initial_frame=frame)
					print(video_inputs_batch.shape)
					print(video_annotations_batch.shape)
					print(video_annotations_indeces_batch.shape)
					y_pred, _ = model(video_inputs_batch, video_inputs_batch[:, :, 0], last_frame)
					print(criterion(y_pred[:, :, video_annotations_indeces_batch[0][0], :, :], video_annotations_batch[:, :, video_annotations_indeces_batch[0][0], :, :]).item())
					frame += 32
					if y_pred_concat.size == 0:
						y_pred_concat = y_pred.cpu().numpy()
					else:
						y_pred_concat = np.concatenate((y_pred_concat, y_pred.cpu().numpy()), axis=2)
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
			mask_for_frames = np.argmax(segs_concat, axis=1)
			mask_for_frames = (mask_for_frames * 255).astype(np.uint8)
			print("finsihed object segmentation...")
			c = Image.fromarray(mask_for_frames, mode='P')
			c.putpalette(Image.ADAPTIVE)
			c.save('test.png', "PNG", mode='P')
			print("segmentation saved...")
	
if __name__ == '__main__':
    inference(config.model_path)