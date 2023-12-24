
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh

def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx

def process_data(data_dir, dataname):
    '''
        功能：检查数据格式、对数据进行格式转换
        参数：
            data_dir : 数据所在目录
            dataname : 数据名称  
    '''
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'): # 检查ply文件
        pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices # 使用trimesh加载数据
        pointcloud = np.asarray(pointcloud) # 转换成numpy数组
    elif os.path.exists(os.path.join(data_dir, dataname) + '.xyz'): # 检查xyz文件
        pointcloud = np.load(os.path.join(data_dir, dataname)) + '.xyz'
    else:
        print('Only support .xyz or .ply data. Please make adjust your data.') # 不支持其他文件
        exit()
    
    # 对点云数据进行预处理
    # 计算点云三个维度点中范围最大的
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    # 计算点云数据的中心点（平均值）
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    # 对点云数据进行归一化处理
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    # 计算点云数据的一些参数
    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = 1000000//POINT_NUM_GT

    # 随机从中选出POINT_NUM_GT个数据
    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud[point_idx,:]
    # print(np.max(pointcloud[:,0]),np.max(pointcloud[:,1]),np.max(pointcloud[:,2]),np.min(pointcloud[:,0]),np.min(pointcloud[:,1]),np.min(pointcloud[:,2]))
    
    # 使用点云数据创建一个KD树 提升近邻搜索的效率
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])
    
    sigmas = np.concatenate(sigmas) # 拉展开数据
    sample = []
    sample_near = []

    # 给点云添加一些噪声模拟真实噪声
    for i in range(QUERY_EACH):
        scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1,POINT_NUM,3)

        sample_near_tmp = [] # 存储每个噪声样本中与原始点云数据最近的点云数据
        for j in range(tt.shape[0]):
            # 找出噪声点在原始点云数据中的最近邻点 将数据转换成GPU上的张量加速运算    
            nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
            nearest_points = pointcloud[nearest_idx]
            nearest_points = np.asarray(nearest_points).reshape(-1,3)
            sample_near_tmp.append(nearest_points)
        sample_near_tmp = np.asarray(sample_near_tmp)
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        sample_near.append(sample_near_tmp)
    
    # 转换为numpy数组 注意全是一维的
    sample = np.asarray(sample)
    sample_near = np.asarray(sample_near)

    # 将预处理好的数据压缩到一起并保存
    np.savez(os.path.join(data_dir, dataname)+'.npz', sample = sample, point = pointcloud, sample_near = sample_near)


class DatasetNP:
    '''

    '''
    def __init__(self, conf, dataname):
        super(DatasetNP, self).__init__()
        '''
        参数：
            conf : dataset配置信息
            dataname : 数据名称
        '''
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir') # 获取数据所在路径
        self.np_data_name = dataname + '.npz' # 添加使用npz文件

        if os.path.exists(os.path.join(self.data_dir, self.np_data_name)): # 文件存在
            print('Data existing. Loading data...')
        else:
            print('Data not found. Processing data...') # 文件不存在
            process_data(self.data_dir, dataname) # 预处理并保存.npz数据

        # 导入数据
        load_data = np.load(os.path.join(self.data_dir, self.np_data_name))
        
        self.point = np.asarray(load_data['sample_near']).reshape(-1,3) # -1表示自动计算维度大小 3表示x、y、z
        self.sample = np.asarray(load_data['sample']).reshape(-1,3)
        self.point_gt = np.asarray(load_data['point']).reshape(-1,3)
        self.sample_points_num = self.sample.shape[0]-1

        # 计算最小值与最大值
        self.object_bbox_min = np.array([np.min(self.point[:,0]), np.min(self.point[:,1]), np.min(self.point[:,2])]) -0.05
        self.object_bbox_max = np.array([np.max(self.point[:,0]), np.max(self.point[:,1]), np.max(self.point[:,2])]) +0.05
        print('Data bounding box:',self.object_bbox_min,self.object_bbox_max)

        # 将numpy数据转换为张量并移动到GPU上
        self.point = torch.from_numpy(self.point).to(self.device).float()       # 噪声点云的最近邻点云 
        self.sample = torch.from_numpy(self.sample).to(self.device).float()     # 加了噪声的点云数据
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float() # 原始点云用于计算损失函数
        
        print('NP Load data: End')

    def np_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse
        points = self.point[index]
        sample = self.sample[index]
        return points, sample, self.point_gt
