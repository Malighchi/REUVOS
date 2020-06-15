import config
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from model import VOSModel
from dataloader import ValidationDataset

def inference(model_path = './SavedModels/folder/model.pth'):
	model = torch.load(model_path)
	model.eval()
	criterion = nn.BCELoss(reduction='mean')
	if config.use_cuda:
		model.cuda()

	with torch.no_grad():
		valid_dataset = ValidationDataset()
		for video in range(len(valid_dataset.valid_frames)):
			for object in range(len(valid_dataset.valid_frames[video])):
				for frame in range(len(valid_dataset.valid_frames[video][object])):
					# Batch-size of model during inference would be 1, because we compute segmentations for 1 video at-a-time
					if(frame == 0):
						video_inputs_batch, video_annotations_batch, video_annotations_indeces_batch = valid_dataset.Datalaoder()
					else:
						frame_nums = 32
						last_frame = video_annotations_batch[:, :, -1]
						if(frame+frame_nums >= len(valid_dataset.valid_frames[video][object])):
							frame_nums = len(valid_dataset.valid_frames[video][object]) - frame
						video_inputs_batch, video_annotations_batch, video_annotations_indeces_batch = valid_dataset.Datalaoder(video_index=video, object_index=object, initial_frame=frame, num_frames=frame_nums)
					y_pred, _ = model(video_inputs_batch, video_inputs_batch[:, :, 0], last_frame)
					print(criterion(y_pred[:, :, video_annotations_indeces_batch[0][0], :, :], video_annotations_batch[:, :, video_annotations_indeces_batch[0][0], :, :]).item())
					frame+=32
					# The model output is (1, 1, 32, 224, 224)
					# Hence, we use the 32nd predicted output frame as 'first_frame_input' during next pass to the model
				# Since, we now have separate segmentation mask for each object np.argmax should help with creating single mask for the entire frame
				# We can save the video-frames using:

				c = Image.fromarray(y_pred, mode='P')
				#c.putpalette(img_palette)
				c.save('frame_name', "PNG", mode='P')
	
if __name__ == '__main__':
    model_path = './SavedModels/Run_2020-06-15 09,33,24/model_1_0.6919.pth'
    inference()