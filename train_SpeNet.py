# -*- coding: utf-8 -*
'''
配置说明：
1. 原位深遥感图像，通过逐像素点除以2047（即2^11)的方式归一化，制作成网络输入，将网络输出保存成图像前，再乘以2047，
复原到原位深图像
2. GF：trainset: 19976 ×（128×128），testset: 90 ×（512×512）
3. QB：trainset: 18123 ×（128×128），testset: 24 ×（512×512）
5. SGD，momentum=0.9， learning_rate 前2层：0.0001，第3层：0.00001
6. MSE loss
7. total epochs： 600， batchsize：128， total_iterations: 1125000
'''
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from torch.nn import init
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import scipy.io as sio
import learnable_SRF
from osgeo import gdal,osr
from os.path import join
import ogr, os


torch.backends.cudnn.enabled = True#使用非确定性算法
torch.backends.cudnn.benchmark = True#自动寻找高效算法
## 超参数设置
version = 1  # 版本号
mav_value = 2047  # GF:1023  QB:2047
satellite = 'QB'  # gf，qb
method = 'SpeNet'
train_batch_size = 64
test_batch_size = 100
total_epochs =50
model_backup_freq=20
num_workers = 1
## 文件夹设置


testsample_dir = '../SpeNet-results/test-samples-v{}/'.format(version)  # 保存测试阶段G生成的图片
evalsample_dir = '../SpeNet-results/eval-samples-v{}/'.format(version)
record_dir = '../SpeNet-results/record-v{}/'.format(version)  # 保存训练阶段的损失值
model_dir = '../SpeNet-results/models-v{}/'.format(version)
backup_model_dir = join(model_dir, 'backup_model/')
checkpoint_model = join(model_dir, '{}-{}-model.pth'.format(satellite, method))
img_dir_MS8=''#MS_PATH
img_dir_PAN32=''#Downsample_PAN_PATH

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
    img = img.astype(np.float32)/ mav_value  # 归一化处理
    return img
def load_panimage(filepath):#输入数据
    img = sio.loadmat(filepath)['imgPAN']  # 原始数据
    img = img.astype(np.float32)/ mav_value  # 归一化处理
    return img

class DatasetFromFolder(Dataset):#数据预加载
    def __init__(self, img_dir_MS8,img_dir_PAN32,transform=None):
        self.img_dir_MS8= img_dir_MS8
        self.input_file1= os.listdir(img_dir_MS8)
        self.img_dir_PAN32 = img_dir_PAN32
        self.input_file4 = os.listdir(img_dir_PAN32)
        self.transform = transform
    def __len__(self):#返回数据集的长度
        return len( self.input_file1 )

    def __getitem__(self, index):  # idx的范围是从0到len（self）根据下标获取其中的一条数据
        input_MS8_path = os.path.join(self.img_dir_MS8, self.input_file1[index])
        input_MS8=load_image(input_MS8_path)

        input_PAN32_path = os.path.join(self.img_dir_PAN32, self.input_file4[index])
        input_PAN32 = load_panimage(input_PAN32_path )


        if self.transform:

            input_MS8= self.transform(input_MS8)
            input_PAN32 = self.transform(input_PAN32)



        return input_PAN32, input_MS8


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

def get_train_set(img_dir_MS8,img_dir_PAN32):
    return DatasetFromFolder(img_dir_MS8,img_dir_PAN32,
                             transform=transforms.Compose([ToTensor()]))


transformed_trainset = get_train_set(img_dir_MS8,img_dir_PAN32)

trainset_dataloader = DataLoader(dataset=transformed_trainset, batch_size=train_batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)


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

#batch_size=一次训练所需要的最大样本
#shuffle=是否打乱训练样本的排列顺序
class MSTPNet(nn.Module):
    def __init__(self):
        super(MSTPNet, self).__init__()
        self.Layer1 = nn.Sequential(
            # Layer1
            nn.Conv2d(in_channels=4, out_channels=20, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(),
        )
        self.Layer2 = nn.Sequential(
            # Layer2
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(),
        )
        self.Layer3 = nn.Sequential(
            # Layer3
            nn.Conv2d(in_channels=40, out_channels=20, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(),
        )
        # Concatanate
        self.Layer4 = nn.Sequential(
            # Layer4
            nn.Conv2d(in_channels=40, out_channels=20, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(),
        )
        # Concatanate
        self.Layer5 = nn.Sequential(
            # Layer5
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(),
        )
        self.Layer6 = nn.Sequential(
            # Layer6
            nn.Conv2d(in_channels=24, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.Tanh()
        )

    def initialize(self):
        for model in self.modules():
            if isinstance(model, nn.Conv2d):
                nn.init.trunc_normal_(model.weight, mean=0.0, std=1e-3)
                nn.init.constant_(model.bias, val=0.0)

    def forward(self, x):
        y1 = self.Layer1(x)
        y2 = self.Layer2(y1)
        x1 = torch.cat([y1, y2], dim=1)
        y3 = self.Layer3(x1)
        x2 = torch.cat([y2, y3], dim=1)
        y4 = self.Layer4(x2)
        y5 = self.Layer5(y4)
        x3 = torch.cat([x, y5], dim=1)
        y6 = self.Layer6(x3)
        return y6
class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()
        self.Layer1 = nn.Sequential(
            # Layer1
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.Layer2 = nn.Sequential(
            # Layer2
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.Layer3 = nn.Sequential(
            # Layer3
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Concatanate
        self.Layer4 = nn.Sequential(
            # Layer4
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Concatanate
        self.Layer5 = nn.Sequential(
            # Layer5
            nn.Conv2d(in_channels=24, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.Layer6 = nn.Sequential(
            # Layer6
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.Tanh()
        )

    def initialize(self):
        for model in self.modules():
            if isinstance(model, nn.Conv2d):
                nn.init.trunc_normal_(model.weight, mean=0.0, std=1e-3)
                nn.init.constant_(model.bias, val=0.0)

    def forward(self, x):
        y1 = self.Layer1(x)
        y2 = self.Layer2(y1)
        y3 = self.Layer3(y2)
        x1 = torch.cat([y1, y3], dim=1)
        y4 = self.Layer4(x1)
        x2 = torch.cat([x, y4], dim=1)
        y5 = self.Layer5(x2)
        y6 = self.Layer6(y5)
        return y6

lr=0.0001
weight_decay=1e-4
xls_path = 'spectral_response.xls'
sp_matrix=learnable_SRF.get_spectral_response(xls_path)
sp_range=learnable_SRF.get_sp_range(sp_matrix)
down_spe=learnable_SRF.convolution_hr2msi(4,1,sp_range).cuda()
criterion =nn.MSELoss().cuda()#定义损失函数
optimizer_down_spe= torch.optim.Adam(down_spe.parameters(),lr=lr,weight_decay=weight_decay)

#SGD属于深度学习的优化方法，类似的优化方法还可以有Adam，momentum等等
if (torch.cuda.device_count() > 1):
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model1,device_ids=[0,1])#用多个GPU加速



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

            out=model1(img_lr_u)

            train_loss =criterion(out,img_pan) #+1-criterion1(out2,img_pan)        #print(train_loss)
            optimizer_down_spe.zero_grad()
            train_loss.backward()   # backpropagation, compute gradients
            optimizer_down_spe.step()
        data = prefetcher.next()
        print('=> {}-{}-Epoch[{}/{}]: train_loss: {:.15f}'.format(satellite, method, epoch, total_epochs, train_loss.item()))
        train_loss_record.write("{:.15f}\n".format( train_loss.item()))
        state = {'down_spe': model1.state_dict(), 'optimizer_down_spe': optimizer_down_spe.state_dict(), 'epoch': epoch}
        torch.save(state, checkpoint_model)  # 保存模型参数，优化器参数
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
    if os.path.exists(checkpoint_model):
        print("==> loading checkpoint '{}'".format(checkpoint_model))
        checkpoint = torch.load(checkpoint_model)
        model1.load_state_dict(checkpoint['down_spe'])
        start_epoch = checkpoint['epoch']
        print('==> 加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('==> 无保存模型，将从头开始训练！')

    train(model, trainset_dataloader, start_epoch)

    #eval(model, testset_dataloader)


if __name__ == '__main__':
    main()
