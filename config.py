from torch import cuda
from datetime import datetime

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

n_epochs = 50
batch_size = 12
n_num_frames = 16

learning_rate = 1e-3
weight_decay = 1e-7

model_id = datetime.now().strftime('%Y-%m-%d %H,%M,%S')
save_dir = './SavedModels/Run_%s/' % model_id

model_path = './SavedModels/modeel_50_0.5433.pth'

