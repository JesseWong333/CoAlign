# 由于nusenes工具库版本的原因，移动到另一库中去
from nuscenes.nuscenes import NuScenes as V2XSimDataset

# use version v1.0-mimi, actually is V2XSimDataset 2.0.  Just to make it compatible
v2x_sim = V2XSimDataset(version='v1.0-mini', dataroot='/data/datasets/V2X-smi/V2X-Sim-2.0', verbose=True)
pass
