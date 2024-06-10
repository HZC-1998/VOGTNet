# -*- coding: utf-8 -*
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
import time
import torch.optim as optim
import scipy.io as sio
import learinable_PSF
import learnable_SRF
import DBFN
from visdom import Visdom
from osgeo import gdal,osr
from os.path import join
import random
import ogr, os

## 超参数设置
version = 1  # 版本号
mav_value = 2047  # GF:1023  QB:2047
satellite = 'QB'  # gf，qb
method = 'DBFN'
train_batch_size =64
test_batch_size = 1
total_epochs = 100
model_backup_freq=20
num_workers = 1
## 文件夹设置

testsample_dir = '../DBFN-results/test-samples-v{}/'.format(version)  # 保存测试阶段G生成的图片
evalsample_dir = '../DBFN-results/eval-samples-v{}/'.format(version)
record_dir = '../DBFN-results/record-v{}/'.format(version)  # 保存训练阶段的损失值
model_dir = '../DBFN-results/models-v{}/'.format(version)
backup_model_dir = join(model_dir, 'backup_model/')
checkpoint_model = join(model_dir, '{}-{}-model.pth'.format(satellite, method))

img_dir_MS8=''#MS_input_path
img_dir_PAN32=''#PAN_input_path
img_dir_MS32=''#reference_input_path

## 创建文件夹
if not os.path.exists(evalsample_dir):
    os.makedirs(evalsample_dir)
if not os.path.exists(testsample_dir):
    os.makedirs(testsample_dir)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(backup_model_dir):
    os.makedirs(backup_model_dir)

## Device configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('==> gpu or cpu:', device, ', how many gpus available:', torch.cuda.device_count())

def load_image(filepath):#输入数据
    img = sio.loadmat(filepath)['imgMS']  # 原始数据
    img = img.astype(np.float32)/ mav_value   # 归一化处理
    return img
def load_panimage(filepath):#输入数据
    img = sio.loadmat(filepath)['imgPAN']  # 原始数据
    img = img.astype(np.float32)/ mav_value   # 归一化处理
    return img
def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
class DatasetFromFolder(Dataset):#数据预加载
    def __init__(self, img_dir_MS8,img_dir_PAN32,img_dir_MS32 ,transform=None):
        self.img_dir_MS8= img_dir_MS8
        self.input_file1= os.listdir(img_dir_MS8)
        self.img_dir_PAN32 = img_dir_PAN32
        self.input_file2 = os.listdir(img_dir_PAN32)
        self.img_dir_MS32=img_dir_MS32
        self.input_file4 = os.listdir(img_dir_MS32)
        self.transform = transform
    def __len__(self):#返回数据集的长度
        return len( self.input_file1 )

    def __getitem__(self, index):  # idx的范围是从0到len（self）根据下标获取其中的一条数据
        input_MS8_path = os.path.join(self.img_dir_MS8, self.input_file1[index])
        input_MS8=load_image(input_MS8_path)

        input_PAN32_path = os.path.join(self.img_dir_PAN32, self.input_file2[index])
        input_PAN32 = load_panimage(input_PAN32_path )

        input_MS32_path = os.path.join(self.img_dir_MS32, self.input_file4[index])
        input_MS32 = load_image( input_MS32_path)

        if self.transform:

            input_MS8= self.transform(input_MS8)
            input_PAN32 = self.transform(input_PAN32)

            input_MS32 = self.transform(input_MS32)

        return input_PAN32, input_MS8, input_MS32


## 变换ToTensor
class ToTensor(object):
    def __call__(self, input):
        # 因为torch.Tensor的高维表示是 C*H*W,所以在下面执行from_numpy之前，要先做shape变换
        # 把H*W*C转换成 C*H*W  4*128*128
        if input.ndim == 3:
            input = np.transpose(input, (2, 0, 1))
            input = torch.from_numpy(input).type(torch.FloatTensor)
        else:
            input = torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor)
        return input

def get_train_set(img_dir_MS8,img_dir_PAN32,img_dir_MS32):
    return DatasetFromFolder(img_dir_MS8,img_dir_PAN32,img_dir_MS32,
                             transform=transforms.Compose([ToTensor()]))


transformed_trainset = get_train_set(img_dir_MS8,img_dir_PAN32,img_dir_MS32)


trainset_dataloader = DataLoader(dataset=transformed_trainset, batch_size=train_batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn,drop_last=True)


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.opt = opt
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch



model = DBFN.UNet()
model = model.cuda()
#定义损失函数
criterion =nn.MSELoss().cuda()



#优化器设置
lr=0.001
WD=1e-4
optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': lr}], lr=lr, weight_decay=WD)
#SGD属于深度学习的优化方法，类似的优化方法还可以有Adam，momentum等等
if (torch.cuda.device_count() > 1):
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model,device_ids=[0,1])#用多个GPU加速
# 模型训练
def train(model, trainset_dataloader, start_epoch):
    print('===>Begin Training!')
    model.train()#启用batch normalization和drop out。
    steps_per_epoch = len(trainset_dataloader)#训练数据个数
    total_iterations = total_epochs * steps_per_epoch#总迭代次数
    train_loss_record = open('%s/train_loss_record.txt' % record_dir, "w")
    epoch_time_record = open('%s/epoch_time_record.txt' % record_dir, "w")
    time_sum = 0

    for epoch in range(start_epoch + 1, total_epochs + 1):
        start = time.time()  # 记录每轮训练的开始时刻
        train_loss = 0.0
        prefetcher = DataPrefetcher(trainset_dataloader)
        data = prefetcher.next()
        i = 0
        while data is not None:
            i += 1
            if i >= total_epochs:
                break
            img_pan,img_lr_u,img= data[0].cuda(),data[1].cuda(),data[2].cuda()# cuda tensor [batchsize,C,W,H]
            output=model(img_lr_u,img_pan)# output of the DBFN
            train_loss =criterion(img,output)
            train_loss.backward()   # backpropagation, compute gradients
            optimizer.step()#反向传播后参数更新
            optimizer_down_spa.step()
        data = prefetcher.next()
        print('=> {}-{}-Epoch[{}/{}]: train_loss: {:.15f}'.format(satellite, method, epoch, total_epochs, train_loss.item()))
        # Save the model checkpoints and backup
        state = {'model': model1.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, checkpoint_model)#保存模型参数，优化器参数
       # state_dict变量存放训练过程中需要学习的权重和偏执系数，state_dict作为python的字典对象将每一层的参数映射成tensor张量，需要注意的是torch.nn.Module模块中的state_dict只包含卷积层和全连接层的参数，当网络中存在batchnorm时，例如vgg网络结构，torch.nn.Module模块中的state_dict也会存放batchnorm's running_mean。
        # backup a model every epoch
        if epoch % model_backup_freq == 0:
           torch.save(model.state_dict(),
                      join(backup_model_dir, '{}-{}-model-epochs{}.pth'.format(satellite, method, epoch)))
        # 输出每轮训练花费时间
        time_epoch = (time.time() - start)
        time_sum += time_epoch
        print('==>No:{} epoch training costs {:.4f}min'.format(epoch, time_epoch / 60))
        epoch_time_record.write(
            "No:{} epoch training costs {:.4f}min\n".format(epoch, time_epoch / 60))
def main():
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    seed_torch(100)
    if os.path.exists(checkpoint_model):
        print("==> loading checkpoint '{}'".format(checkpoint_model))
        checkpoint = torch.load(checkpoint_model)
        model1.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('==> 加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('==> 无保存模型，将从头开始训练！')

    train(model, trainset_dataloader, start_epoch)


if __name__ == '__main__':
    main()
