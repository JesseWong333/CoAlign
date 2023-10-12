import os
import json

def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
        return my_json

def write_json(path_json, data):
    with open(path_json, "w") as f:
        json.dump(data, f)

def idx_batch_mapping(inf_data_infos):
    idx_batch_mappings = {}
    for inf_data_info in inf_data_infos:
        inf_idx = inf_data_info['pointcloud_path'].split('/')[-1].replace('.pcd', '')
        idx_batch_mappings[inf_idx] = inf_data_info['batch_id']
    return idx_batch_mappings

# def get_all_inf_ids(co_datainfo):
#     infrastructure_idxs = set() # should be set
#     for data_info in co_datainfo:
#         inf_idx = data_info['infrastructure_pointcloud_path'].split('/')[-1].replace('.pcd', '')
#         infrastructure_idxs.add(inf_idx)
#     return infrastructure_idxs

def get_all_inf_ids(inf_data_infos):
    infrastructure_idxs = {} # should be set
    for data_info in inf_data_infos:
        inf_idx = data_info['pointcloud_path'].split('/')[-1].replace('.pcd', '')
        infrastructure_idxs[inf_idx] = data_info
    return infrastructure_idxs

def generate_flow(co_datainfo, inf_datainfo, split_train, split_val):
    # find max=5 the history frames; save it back to co_datainfo
    # we dont need to split the co_datainfo data; 
    inf_idx_batch_mappings = idx_batch_mapping(inf_datainfo)  # inf_idx： batch_id
    inf_idxs = get_all_inf_ids(inf_datainfo)

    for data_info in co_datainfo:
        # find max=5 previous frames
        inf_idx = data_info['infrastructure_pointcloud_path'].split('/')[-1].replace('.pcd', '')
        for i in range(1, 6):
            previous_id = str(int(inf_idx) - i).zfill(6)
            if previous_id not in inf_idxs or inf_idx_batch_mappings[previous_id] != inf_idx_batch_mappings[inf_idx]:
                data_info["previous_inf_" + str(i)] = None
            else:
                data_info["previous_inf_" + str(i)] = ("infrastructure-side/velodyne/" + previous_id + ".pcd", 
                                                       (int(inf_idxs[inf_idx]['pointcloud_timestamp']) - int(inf_idxs[previous_id]['pointcloud_timestamp'])) /1000 )  # 记录path和时延
    
    return co_datainfo

if __name__ == "__main__":
    data_dir = "/data/datasets/DAIR-V2X/cooperative-vehicle-infrastructure"
    train_dir = "/data/datasets/DAIR-V2X/cooperative-vehicle-infrastructure/train.json"
    validate_dir = "/data/datasets/DAIR-V2X/cooperative-vehicle-infrastructure/val.json"
    
    co_datainfo = read_json(os.path.join(data_dir, 'cooperative/data_info.json'))  # 共6617帧
    inf_datainfo = read_json(os.path.join(data_dir, 'infrastructure-side/data_info.json'))  # 共12424帧

    split_train = read_json(train_dir)  # using vehicle frame ID as the key
    split_val = read_json(validate_dir)

    co_datainfo = generate_flow(co_datainfo, inf_datainfo, split_train, split_val)
    
    write_json(os.path.join(data_dir, 'cooperative/data_info_with_delay.json'), co_datainfo)


