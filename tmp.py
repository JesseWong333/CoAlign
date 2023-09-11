import torch
import torch.nn as nn
torch.use_deterministic_algorithms(True)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
set_seed(42)

m = nn.ConvTranspose2d(16, 33, 3, stride=2)

input = torch.randn(20, 16, 50, 100)
output = m(input)
pass
