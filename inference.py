import config
import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
from model import VOSModel
from dataloader import ValidationDataset
from torch.utils.data import DataLoader

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
		dloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
		print('dataset loaded...')
		for i, sample in enumerate(dloader):
			video_inputs_batch, video_annotations_batch, video_indeces_batch = sample
			video_indeces_batch = video_indeces_batch.cpu().detach().numpy()
			if config.use_cuda:
				video_inputs_batch = video_inputs_batch.cuda()
				video_annotations_batch = video_annotations_batch.cuda()
			segs_concat = np.array([])
			for object in range(len(valid_dataset.valid_frames[i])):
				start_frame = 0
				num_frames = 32
				y_pred_concat = np.array([])
				while start_frame < len(valid_dataset.valid_frames[i][object]):
					if(start_frame == 0):
						start_frame = int(video_indeces_batch[0][0][object])
						last_frame = video_annotations_batch[:, :, object]
						if start_frame > len(y_pred_concat):
							y_pred_concat = torch.from_numpy(np.zeroes((1, 1, start_frame, 224, 224)))
					else:
						last_frame = y_pred[:, :, -1]
					if (start_frame + num_frames) >= len(valid_dataset.valid_frames[i][object]):
						num_frames = len(valid_dataset.valid_frames[i][object]) - start_frame
					frame_selection = range(start_frame, start_frame+num_frames)
					while len(frame_selection) < 32:
						frame_selection.append(frame_selection[-1])
					y_pred, _ = model(video_inputs_batch[:, :, frame_selection, :, :], video_inputs_batch[:, :, start_frame], last_frame)
					#loss = criterion(y_pred[:, :, video_annotations_indeces_batch[0][0], :, :], video_annotations_batch[:, :, video_annotations_indeces_batch[0][0], :, :])
					#print(loss.item())
					start_frame += 32
					if y_pred_concat.size == 0:
						y_pred_concat = torch.round(y_pred).cpu().numpy()
					else:
						y_pred_concat = np.concatenate((y_pred_concat, torch.round(y_pred).cpu().numpy()), axis=2)
				frames_to_use = range(len(valid_dataset.valid_frames[i][0]))
				y_pred_concat = y_pred_concat[:, :, frames_to_use, :, :]
				if segs_concat.size == 0:
					test_ann = (np.ones((y_pred_concat.shape)) *.000009)
					segs_concat = np.concatenate((test_ann, y_pred_concat), axis=1)
				else:
					segs_concat = np.concatenate((segs_concat, y_pred_concat), axis=1)

			segs_concat = (segs_concat.squeeze(0))
			mask_for_frames = np.argmax(segs_concat, axis=0).astype(dtype=np.uint8)
			print("finsihed object segmentation...")
			video_file = valid_dataset.valid_frames[i][0][0].split('/')
			try:
				dir = 'Annotations/%s/' % video_file[-2]
				os.mkdir(dir)
			except:
				pass
			palette = Image.open(valid_dataset.valid_annotations[i][0][0])
			for images in range(len(valid_dataset.valid_frames[i][0])):
				img_file = valid_dataset.valid_frames[i][0][images].split('/')
				c = Image.fromarray(mask_for_frames[images], mode='P').resize(size=palette.size)
				c.putpalette(palette.getpalette())
				img_path = dir + img_file[-1]
				img_path = img_path.replace('jpg', 'png')
				c.save(img_path, "PNG", mode='P')
			print("segmentation saved...")
	
if __name__ == '__main__':
	inference(config.model_path)