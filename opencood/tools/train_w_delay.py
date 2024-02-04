# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# modified by Junjie Wang <junjie.wang@umu.se>

import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from collections import OrderedDict
import glob
from icecream import ic
import random
import numpy as np
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
set_seed(42)

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2
                              ) 
    # val loader 重新构建
    max_time_delay = hypes['time_delay']

    val_loaders = []
    for time_delay in range(max_time_delay+1):
        hypes.update({"time_delay": time_delay})
        opencood_validate_dataset = build_dataset(hypes,
                                                visualize=False,
                                                train=False)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=2)
        val_loaders.append(val_loader)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if hypes['train_stage'] == 'stage1':
        # train the detection model
        for name, value in model.named_parameters():
            if name.startswith('meta_flow'):
                value.requires_grad = False
            else:
                value.requires_grad = True
        # setup optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = train_utils.setup_optimizer(hypes, params)
    elif hypes['train_stage'] == 'stage2':
        assert 'stage1_model' in hypes
        main_model_dict = torch.load(hypes['stage1_model'])
        model.load_state_dict(main_model_dict, strict=False) # missing meta_flow model; 
        # train the flow model with a trained detection model
        for name, value in model.named_parameters():
            if name.startswith('meta_flow'):
                value.requires_grad = True # Note, the BN of is still updating
            else:
                value.requires_grad = False
        # setup optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = train_utils.setup_optimizer(hypes, params)
    else:
        # optimizer setup
        optimizer = train_utils.setup_optimizer(hypes, model)

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)
    
    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            # the model will be evaluation mode during validation
            model.train()
            if hypes['train_stage'] == 'stage2':
                model.eval()  # we call eval(), just inorder to avoid the norm layer update
                model.meta_flow.train() 
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag and hypes['train_stage'] == 'stage1':
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()

            # torch.cuda.empty_cache()

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for val_loader in tqdm(val_loaders):
                    for i, batch_data in enumerate(val_loader):
                        if batch_data is None:
                            continue
                        model.zero_grad()
                        optimizer.zero_grad()
                        model.eval()

                        batch_data = train_utils.to_device(batch_data, device)
                        batch_data['ego']['epoch'] = epoch
                        ouput_dict = model(batch_data['ego'])

                        final_loss = criterion(ouput_dict,
                                            batch_data['ego']['label_dict'])
                        valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = True    
    # ddp training may leave multiple bestval
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
    
    if len(bestval_model_list) > 1:
        import numpy as np
        bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])

    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference_w_delay.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()  