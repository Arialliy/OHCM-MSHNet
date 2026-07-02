import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['ACM', 'ALCNet','DNANet', 'ISNet', 'RDIAN', 'ISTDU-Net', 'HCNet'], nargs='+',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet', 'HCNet', 'MSHNet'")
parser.add_argument("--pth_dirs", default=None, nargs='+',  help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'], nargs='+', 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--lambda_hc", type=float, default=0.0)
parser.add_argument("--hc_topk_ratio", type=float, default=0.01)
parser.add_argument("--hc_dilate_kernel", type=int, default=7)
parser.add_argument("--hc_gamma", type=float, default=2.0)
parser.add_argument("--hc_warm_epoch", type=int, default=10)
parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
parser.add_argument("--mshnet_in_channels", type=int, default=1)

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)
  
def test(): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(
        model_name=opt.model_name,
        mode='test',
        loss_cfg=vars(opt),
    ).to(device)
    try:
        net.load_state_dict(torch.load(opt.pth_dir, map_location=device)['state_dict'])
    except:
        net.load_state_dict(torch.load(opt.pth_dir, map_location=device)['state_dict'])
    net.eval()

    eval_metrics = BinaryMetricsGPU(threshold=opt.threshold, device=device)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            img = Variable(img).to(device)
            gt_mask = gt_mask.to(device)
            pred = net.forward(img)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            pred = pred[:, :, :h, :w]
            gt_mask = gt_mask[:, :, :h, :w]
            eval_metrics.update(pred, gt_mask)
            if (idx_iter + 1) % 100 == 0:
                print(f"Test [{idx_iter + 1}/{len(test_loader)}]", flush=True)
            
            ### save img
            if opt.save_img == True:
                img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')  
    
    metric_results = eval_metrics.get()
    params_count = Params(net)

    result_lines = [
        "pixAcc:\t" + str(metric_results['pixAcc']),
        "mIoU:\t" + str(metric_results['mIoU']),
        "nIoU:\t" + str(metric_results['nIoU']),
        "PixelRecall:\t" + str(metric_results['PixelRecall']),
        "Fa:\t" + str(metric_results['Fa']),
        "Precision:\t" + str(metric_results['Precision']),
        "Recall:\t" + str(metric_results['Recall']),
        "F1:\t" + str(metric_results['F1']),
        "Params:\t%s (%d)" % (format_params(params_count), params_count),
    ]
    for line in result_lines:
        print(line, flush=True)
        opt.f.write(line + '\n')
    opt.f.flush()

    try:
        flops_count = FLOPs(net, input_size=(1, 1, 256, 256), device=device)
        flops_line = "FLOPs:\t%s (%d)" % (format_flops(flops_count), flops_count)
    except Exception as exc:
        flops_line = "FLOPs:\tUnavailable (%s)" % exc
    print(flops_line, flush=True)
    opt.f.write(flops_line + '\n')
    opt.f.flush()

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
        
