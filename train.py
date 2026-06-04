from data_process import build_train_loaders, build_test_loaders
from model import DRp_single_pan_and_ms_share_t
from torch.optim import Adam
from rectified_flow import RectifiedFlow
import torch
import os
import sys         
import time        
import numpy as np
from metricsm import get_metrics_reduced
from evall import *
from set_random import set_seed
from datetime import datetime, timedelta

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"runs/train_{timestamp}"
weight_dir = os.path.join(run_dir, "weights")
result_dir = os.path.join(run_dir, "results")
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
train_path = ["/data0/train_gf2"]
train_loader_list = build_train_loaders(train_path=train_path,max_value=[1023.0],batch_size=32)
test_path = ["/data0/test_gf2_r_20"]
test_loader_list = build_test_loaders(train_path=test_path,max_value=[1023.0],batch_size=1,shuffle=False,is_my=False,flag=0)
net = DRp_single_pan_and_ms_share_t(dim=[64,64*2,64*4]).to(device)
rf = RectifiedFlow(net,num_timesteps=1000).to(device)
set_seed(42)
optimizer = Adam(net.parameters(),1e-4,betas=(0.9,0.999))

def write_list_to_txt(lst, filename):
    with open(filename, 'a') as file:
        for item in lst:
            file.write(str(item) + '\n')

if os.path.exists('gf2_less_cross/-1.pth'):
    print('RF net is loading parameters')
    net.load_state_dict(torch.load('gf2_less_cross/-1.pth',map_location=device))

step = 0
start = 0
epochs = 3000
start_eval = 0
best_psnr = 0
best_epoch = 0

date_str = datetime.now().strftime("%m-%d") 
model_name = net.__class__.__name__

log_filename = f"{date_str}-{model_name}-train.txt"
log_file = os.path.join(run_dir, log_filename)

with open(log_file, 'w') as f:
    f.write(f"=== 实验启动信息 ===\n")
    f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"模型名称: {model_name}\n")
    f.write(f"====================\n\n")

print(f"日志文件已创建: {log_file}")
prev_time = time.time()

for epoch in range(start,epochs):
    net.train()
    print(f'\nTraining for dataset: {train_path[0]}')
    train_loader = train_loader_list[0]
    epoch_loss = []
    
    for index, train_data in enumerate(train_loader):
        pan, lrms, hrms, ms = train_data
        pan = pan.to(device)
        lrms = lrms.to(device)
        hrms = hrms.to(device)

        optimizer.zero_grad()

        loss = rf(hrms-lrms,pan,lrms)
        loss.backward()
        optimizer.step()
        current_loader_len = len(train_loader)
        batches_done = (epoch - start) * current_loader_len + index
        batches_left = (epochs - start) * current_loader_len - batches_done
        
        time_elapsed = time.time() - prev_time
        time_left = timedelta(seconds=int(batches_left * time_elapsed))
        prev_time = time.time() 

        sys.stdout.write(
            "\rshare_t_b2 Step:%d Epoch:%d/%d Batch:%d/%d Loss:%.4f best_ep:%d best_psnr:%.4f ETA:%s " %
            (step, epoch, epochs, index + 1, current_loader_len, loss.item(), best_epoch, best_psnr, str(time_left))
        )
        sys.stdout.flush()

        
        step = step + 1
        
    print() 
        
    if start_eval <= epoch or epoch < 6:
        net.eval()
        print('Test')
        psnr_loss,ssim,cc,sam,ergas = [],[],[],[],[]
        train_loader = test_loader_list[0]
        
        epoch_img_dir = os.path.join(result_dir, f"epoch_{epoch}")
        os.makedirs(epoch_img_dir, exist_ok=True)
        
        for index, train_data in enumerate(train_loader):
            pan, lrms, hrms, ms = train_data
            pan = pan.to(device)
            lrms = lrms.to(device)
            hrms = hrms.to(device)
            with torch.no_grad():
                out = rf.sample(pan,lrms,sample_steps=1)
                output = out[-1] + lrms
                output_clamped = output.clamp(0.0,1.0)
                
                m1,m2,m3,m4,m5 = get_metrics_reduced(output_clamped, hrms.to(device))
                psnr_loss.append(m1)
                ssim.append(m2)
                cc.append(m3)
                sam.append(m4)
                ergas.append(m5)
    
        psnr_mean = np.asarray(psnr_loss).mean()
        ssim_mean = np.asarray(ssim).mean()
        cc_mean = np.asarray(cc).mean()
        sam_mean = np.asarray(sam).mean()
        ergas_mean = np.asarray(ergas).mean()
        print(f"[*] Test Results -> PSNR: {psnr_mean:.4f}, SSIM: {ssim_mean:.4f}, CC: {cc_mean:.4f}, SAM: {sam_mean:.4f}, ERGAS: {ergas_mean:.4f}")
        print("=" * 80)
        
        epoch_loss.append((epoch,psnr_mean, ssim_mean, cc_mean, sam_mean, ergas_mean))
        rounded_epoch_loss = [
            tuple(round(float(val), 8) for val in loss) for loss in epoch_loss
        ]
        write_list_to_txt(rounded_epoch_loss, log_file)
        
        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            best_epoch = epoch
            print(f'New Best PSNR: {best_psnr:.4f} at epoch {best_epoch}, saving model...')
            
            weight_name = os.path.join(weight_dir, f"best_epoch{epoch}_psnr{best_psnr:.4f}.pth")
            torch.save(net.state_dict(), weight_name)
            
print('Final best psnr:', best_psnr)
