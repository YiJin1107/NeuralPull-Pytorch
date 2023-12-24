# -*- coding: utf-8 -*-
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import DatasetNP
from models.fields import NPullNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log
import math
import mcubes
import warnings
warnings.filterwarnings("ignore")


class Runner:
    '''
    负责加载配置文件、创建模型、设置优化器
    '''

    def __init__(self, args, conf_path, mode='train'):
        '''
        参数：
            args : 命令行参数，指定各种配置
            conf_path : 配置文件的路径
            mode : 选择运行模式

        '''

        self.device = torch.device('cuda') # 设置使用GPU加速

        # 基础配置
        self.conf_path = conf_path # 配置文件路径
        f = open(self.conf_path)
        conf_text = f.read() # 读取配置文件内容
        f.close()

        # 创建模型输出文件夹
        self.conf = ConfigFactory.parse_string(conf_text)                                   # 解析配置文件为Config对象
        # self.conf['dataset.np_data_name'] = self.conf['dataset.np_data_name']
        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True) 
        
        
        self.dataset_np = DatasetNP(self.conf['dataset'], args.dataname)                    # 创建数据集对象 加载和预处理数据
        self.dataname = args.dataname                                                       # 初始化数据名称
        self.iter_step = 0                                                                  # 初始化迭代步长

        # 训练参数
        self.maxiter = self.conf.get_int('train.maxiter')                                   # 最大迭代次数
        self.save_freq = self.conf.get_int('train.save_freq')                               # 模型保存频率
        self.report_freq = self.conf.get_int('train.report_freq')                           # 获取训练报告频率
        self.val_freq = self.conf.get_int('train.val_freq')                                 # 获取验证的概率
        self.batch_size = self.conf.get_int('train.batch_size')                             # 训练batch大小
        self.learning_rate = self.conf.get_float('train.learning_rate')                     # 学习率
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)            # 预热结束时间
        self.eval_num_points = self.conf.get_int('train.eval_num_points')                   # 评估点的数量

        self.mode = mode


        # 创建一个神经网络对象，并将其移动到GPU上。神经网络对象用于实现模型的信息前向传播和误差反向传播
        self.sdf_network = NPullNetwork(**self.conf['model.sdf_network']).to(self.device)
        # 创建一个优化器用于优化神经网络
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)

        # 备份代码并debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())                        # 获取当前日期
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')        # 构建日志文件日志
        logger = get_root_logger(log_file=log_file, name='outs')                            # 创建一个日志记录器用于记录训练信息
        self.logger = logger
        batch_size = self.batch_size                                                        # 定义训练批的大小

        res_step = self.maxiter - self.iter_step                                            # 剩余迭代次数

        for iter_i in tqdm(range(res_step)):                                                # 用tqdm库创建一个进度条迭代res_step次
            self.update_learning_rate_np(iter_i)                                            # 更新学习率
            # 最近点云、噪声点云、原始点云
            points, samples, point_gt = self.dataset_np.np_train_data(batch_size)   # 从数据集中获取训练数据 
                
            samples.requires_grad = True
            gradients_sample = self.sdf_network.gradient(samples).squeeze()         # 5000x3 计算梯度
            sdf_sample = self.sdf_network.sdf(samples)                              # 5000x1 计算SDF值
            grad_norm = F.normalize(gradients_sample, dim=1)                        # 5000x3 对梯度归一化
            sample_moved = samples - grad_norm * sdf_sample                         # 5000x3 调整采样数据的位置

            loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean() # SDF损失函数
            
            loss = loss_sdf
            
            self.optimizer.zero_grad()                                      # 将优化器的梯度缓冲区清零
            loss.backward()                                                 # 计算损失的梯度
            self.optimizer.step()                                           # 更新模型参数
            
            self.iter_step += 1
            # 打印迭代步数、SDF损失和学习率的日志信息
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} cd_l1 = {} lr={}'.format(self.iter_step, loss_sdf, self.optimizer.param_groups[0]['lr']), logger=logger)

            # 验证网路有效性
            if self.iter_step % self.val_freq == 0 and self.iter_step!=0: 
                self.validate_mesh(resolution=256, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger)

            # 保存检查点
            if self.iter_step % self.save_freq == 0 and self.iter_step!=0: 
                self.save_checkpoint() 


    def validate_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None):
        """
        用给定的阈值生成网格
        """
        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network.sdf(pts))

        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))


    def update_learning_rate_np(self, iter_step):
        """
        更新学习率
        """
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        """
        提取场
        参数：
            bound_min : x y z 轴的最小值
            bound_max : x y z 轴的最大值
            resolution : 每个轴上划分网格的数量 
        """
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u
    
    
    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh

    # 备份保存文件
    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    # 导入模型
    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
    
    # 保存模型
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')                 # 设置张量为浮点型
    parser = argparse.ArgumentParser()                                      # 创建参数解析器对象
    parser.add_argument('--conf', type=str, default='./confs/np_srb.conf')  # 指定配置文件路径
    parser.add_argument('--mode', type=str, default='train')                # 规定运行模式
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)      # 指定阈值
    parser.add_argument('--gpu', type=int, default=0)                       # 指定GPU
    parser.add_argument('--dir', type=str, default='gargoyle')              # 指定数据输出目录名称
    parser.add_argument('--dataname', type=str, default='gargoyle')         # 指定数据名称
    args = parser.parse_args()                                              # 解析命令行参数并保存在args中

    torch.cuda.set_device(args.gpu)                                         # 设定使用哪个GPU进行计算
    runner = Runner(args, args.conf, args.mode)

    if args.mode == 'train':            # 训练模式
        runner.train()
    elif args.mode == 'validate_mesh':  # 生成点云数据
        runner.load_checkpoint('ckpt_040000.pth')  # 加载训练过的网络
        # 设定的阈值
        threshs = [-0.001,-0.0025,-0.005,-0.01,-0.02,0.0,0.001,0.0025,0.005,0.01,0.02]
        for thresh in threshs:
            runner.validate_mesh(resolution=256, threshold=thresh)