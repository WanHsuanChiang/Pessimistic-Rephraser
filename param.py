import torch

MAX_LEN = 120
MODEL = 'xlnet-base-cased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#n_gpu = torch.cuda.device_count()
#torch.cuda.get_device_name(0)