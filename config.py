from torch import cuda
from datetime import datetime

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

n_epochs = 100
batch_size = 12
n_frames = 32
skew_weight = False
n_anns = 7

learning_rate = 1e-3
weight_decay = 1e-7

model_id = datetime.now().strftime('%Y-%m-%d %H,%M,%S')
save_dir = './SavedModels/Run_%s/' % model_id

model_path = './SavedModels/fix_model64_30_0.5417.pth'

bce_w_logits = True
use_hidden_state = False
use_fixes = False
