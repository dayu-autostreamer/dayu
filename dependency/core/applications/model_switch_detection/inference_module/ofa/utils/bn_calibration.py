from models.ops.dynamic_batchnorm2d import DynamicBatchNorm2d
from models.utils import AverageMeter, get_net_device
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_running_statistics(model, image_loader, max_iters=10):
    # 对于bn测量，可以用gpu加速
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    bn_mean = {}
    bn_var = {}
    forward_model = copy.deepcopy(model)    
    forward_model.eval()

    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = (
                        x.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = (
                        batch_var.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    if len(bn_mean) == 0:
        # skip if there is no batch normalization layers in the network
        return

    with torch.no_grad():
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        cur_iter = 0
        for image in image_loader:
            image = image.to(get_net_device(forward_model))
            forward_model(image)
            cur_iter += 1
            if(cur_iter > max_iters):
                break
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

def save_bn_statistics(model, save_path):
    """
    保存模型中所有BN层的running_mean和running_var到文件
    
    Args:
        model: 神经网络模型
        save_path: 保存统计数据的文件路径(.pth)
    """
    bn_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, DynamicBatchNorm2d)):
            bn_stats[name] = {
                'running_mean': m.running_mean.data.cpu().numpy(),
                'running_var': m.running_var.data.cpu().numpy()
            }
    
    torch.save(bn_stats, save_path)
    print(f"BN statistics saved to {save_path}")

def load_bn_statistics(model, stats_path):
    """
    从文件加载BN统计数据并应用到模型
    
    Args:
        model: 神经网络模型
        stats_path: 包含BN统计数据的文件路径(.pth)
    """
    bn_stats = torch.load(stats_path)
    
    for name, m in model.named_modules():
        if name in bn_stats and isinstance(m, (nn.BatchNorm2d, DynamicBatchNorm2d)):
            stats = bn_stats[name]
            feature_dim = stats['running_mean'].shape[0]
            
            # 将统计数据加载到模型中
            m.running_mean.data[:feature_dim].copy_(torch.from_numpy(stats['running_mean']).to(m.running_mean.device))
            m.running_var.data[:feature_dim].copy_(torch.from_numpy(stats['running_var']).to(m.running_var.device))
    
    print(f"BN statistics loaded from {stats_path}")