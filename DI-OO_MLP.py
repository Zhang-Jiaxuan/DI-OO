# sample possible world
# build_space
from training.preprocess import preprocess
import numpy as np
# from experiment import run_random, run_classic_clean, run_boost_clean, run_cp_clean
import utils
import copy
import argparse
import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import higher
import random
import itertools
from sklearn.metrics import roc_curve, auc,roc_auc_score
import time

###########固定随机种子#######################
seed = 1  # seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed)#numpy产生的随机数一致
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


#####################模型############################
# X=df.drop(columns=result_var)
# print(X)
class tabularDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


# train_ds = tabularDataset(X, Y)
# val_ds = tabularDataset(X_val, Y_val)


# print(train_ds[0])


class tabularModel(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, 500)
        self.lin2 = nn.Linear(500, 100)
        self.lin3 = nn.Linear(100, 2)
        self.bn_in = nn.BatchNorm1d(input_size)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x_in):
        # print(x_in.shape)
        x = self.bn_in(x_in)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        # print(x)

        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        # print(x)

        x = self.lin3(x)
        x = torch.sigmoid(x)
        return x

import math
def index_max(x):
    """
    获取向量x的最大值的索引
    """
    m = max(x)
    return x.index(m)


class UCB1():
    def __init__(self, counts, values,beta):
        self.counts = counts
        self.values = values
        self.beta=beta
        return

    def initialize(self, n_arms):  # 初始化有多少个臂
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        n_arms = len(self.counts)
        # 确保所有的臂都至少玩了一次
        # 从而可以对所有可用的臂有一个初始化的了解
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            # 使用置信区间上界
            # 置信度为1-2/total_counts
            bonus = math.sqrt((2 * math.log(total_counts)) /
                              float(self.counts[arm]))
            # ucb_values[arm] = self.values[arm] + bonus
            ucb_values[arm] = self.values[arm]*self.beta + bonus#调整参数

        return index_max(ucb_values)  # 返回ucbvalue最大的索引

    def update(self, chosen_arm, reward):  # 更新被选中的那个arm的value
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value  # 更新arm
        return
#################数据##########################
def make_space(X_train_repairs, X_val, gt=None):
    space_X = np.transpose(X_train_repairs, (1, 0, 2))  ## shape (#row, #repair, #column)
    space_X = np.around(space_X, decimals=12)
    gt = np.around(gt, decimals=12)
    # for s in space_X:
    #     l = len(s)
    #     if (l != 1):
    #         if (l != 10):
    #             print("space0 error", l)
    space = []
    gt_indices = []
    isTrue = []
    # print('------------isTrue-----------------------')
    for X, X_gt in zip(space_X, gt):
        # zip()是Python的一个内建函数
        X_unique, X_indices = np.unique(X, axis=0, return_index=True)
        # 对于一维数组或者列表，np.unique () 函数 去除其中重复的元素 ，np.unique(a, axis=0) 去除重复行
        # return_index：如果为 true，返回新列表元素在旧列表中的位置（下标），并以列表形式存储
        # print(X_indices)如果len=1说明原本没有缺失
        if (len(X_indices) <= 1):
            isTrue.append(1)
            space.append(X_unique)
        else:
            isTrue.append(0)  # len>1说明
            space.append(X)
        # S_unique = S[X_indices]#相似度中被选择的哪个，即被选中元素的再旧列表中的位置下标
        # print((X_unique == X_gt).all(axis=1))输出和ground_truth相同的那一行
        gt_id = np.argwhere((X == X_gt).all(axis=1))[0, 0]
        # gt_id = np.argwhere((X_unique == X_gt).all(axis=1))[0, 0]
        # np.all(arr,axis=1)，表示在水平方向上计算元素的与运算（AND）
        # #为了让space中的脏行的候选集数量相同
        # S_val_t.append(S_unique)
        gt_indices.append(gt_id)  # 输出每行数据和ground_truth相同的那一行
    # print(isTrue)
    return space, gt_indices, isTrue


def select_clean(space, gt_indices, indices,isTrue, labels):
    # 选出所有干净行，干净行跟label对上，然后不需要gt_indices，分完之后也不需要istrue了
    # clean_list
    clean_rows = []
    clean_rows_labels = []
    dirty_rows = []
    dirty_rows_labels = []
    new_gt_indices = []
    clean_indices=[]
    dirty_indices=[]



    for i, s, gt, indice,l in zip(isTrue, space, gt_indices,indices, labels):
        if i == 1:  # 此行干净，加入clean_rows
            clean_rows.append(s.tolist()[0])
            clean_rows_labels.append(l)
            clean_indices.append(indice)
        else:  #
            dirty_rows.append(s.tolist())
            new_gt_indices.append(gt)
            dirty_rows_labels.append(l)
            dirty_indices.append(indice)
    return clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices,clean_indices,dirty_indices


# 先完全随机sample，不管它时属于哪个数据集的。
def sample_by_confidence2(model, train_dataset, confidence_threshold, sample_size):
    """
    基于训练集置信度进行样本抽样
    :param model: 已训练好的模型对象
    :param train_dataset: 训练集数据对象
    :param confidence_threshold: 置信度阈值，小于该阈值的样本会被抽取
    :param sample_size: 抽样的样本数量
    :return: 返回被抽取样本的位置列表和被抽取的训练集数据
    """
    positions = []
    samples = []

    # 遍历训练集中的每个样本
    for i in range(len(train_dataset)):
        x, y_true = train_dataset[i]
        # 将输入数据转换为模型可接受的形式（例如Tensor）
        x = torch.tensor(x).unsqueeze(0)
        # 将输入数据通过模型得到预测结果
        y_pred = model(x).squeeze().item()
        # 计算预测结果的置信度
        confidence = abs(y_pred - y_true)

        # 如果置信度小于阈值，则将该样本加入抽样列表
        if confidence < confidence_threshold:
            positions.append(i)
            samples.append((x, y_true))

        # 如果已经抽取到足够数量的样本，则退出循环
        if len(samples) >= sample_size:
            break

    return positions, samples


def sample_by_confidence(model, train_data_2d, num_samples):
    # print(train_data_2d)
    # print(len(train_data_2d))
    model.eval()
    y = pd.DataFrame([0 for i in range(len(train_data_2d))])
    dirty_label = LabelEncoder()
    y = dirty_label.fit_transform(y)
    x = pd.DataFrame(train_data_2d)
    print(x)
    print(y)
    dirty_ds = tabularDataset(x, y)
    dirty_dl = DataLoader(dirty_ds, batch_size=len(train_data_2d), shuffle=False)
    with torch.no_grad():
        for i, (x, y) in enumerate(dirty_dl):
            x = x.float().to(DEVICE)
            outputs = model(x).cpu()
            confidences, _ = torch.max(outputs.float(), dim=1)
    print(confidences)
    # 按照置信度sample训练集
    weights = confidences / confidences.sum()
    sampled_indices = torch.multinomial(weights, num_samples=num_samples)
    sample_data = train_data_2d[sampled_indices]
    # sampled_indices=sampled_indices.reshape(train_data.shape[:2])
    return sample_data, sampled_indices


def random_sample(dirty_rows):  # 随机性控制在
    N = len(dirty_rows)
    M = len(dirty_rows[0])
    sample_train_data = []
    # rd = np.random.RandomState(888)
    random_id_list = np.random.randint(0, M, N)  # 因为要确定随机性，所以不能每次都是一样的随机数，不能确定随机顺序，不可复现？
    for row, rd_id in zip(dirty_rows, random_id_list):
        sample_train_data.append(row[rd_id])
    return sample_train_data, random_id_list


def cleanByIndex(index_to_clean, clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices,
                 dirty_matrix,clean_indices,dirty_indices):
    # print('clean_rows')
    # print(clean_rows)
    # print('len(clean_rows)')
    # print(len(clean_rows))
    # print(type(clean_rows))
    # print(type(clean_rows[1]))
    # print(type(dirty_rows[index_to_clean][new_gt_indices[index_to_clean]]))
    # print(type(clean_rows[-1]))
    # clean_rows.append(dirty_rows[index_to_clean][new_gt_indices[index_to_clean]] if isinstance(dirty_rows[index_to_clean][new_gt_indices[index_to_clean]],list) else dirty_rows[index_to_clean][new_gt_indices[index_to_clean]].tolist())
    cur_indice=dirty_indices[index_to_clean]
    clean_indices.append(cur_indice)
    dirty_indices=np.delete(dirty_indices,index_to_clean)
    clean_rows.append(dirty_rows[index_to_clean][new_gt_indices[index_to_clean]])
    clean_rows_labels.append(dirty_rows_labels[index_to_clean])
    dirty_matrix = np.delete(dirty_matrix, index_to_clean, axis=0)
    new_dirty_rows = np.delete(dirty_rows, index_to_clean, axis=0)
    new_dirty_rows_labels = np.delete(dirty_rows_labels, index_to_clean)
    new_gt_indices = np.delete(new_gt_indices, index_to_clean)
    return clean_rows, clean_rows_labels, new_dirty_rows, new_dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices


def deleteByIndex(dirty_rows, dirty_rows_labels, index_to_delete, new_gt_indices):
    # dirty_matrix = np.delete(dirty_matrix, index_to_delete, axis=0)
    new_dirty_rows = np.delete(dirty_rows, index_to_delete, axis=0)  # 可以一下去掉很多行
    new_dirty_rows_labels = np.delete(dirty_rows_labels, index_to_delete)
    new_gt_indices = np.delete(new_gt_indices, index_to_delete)
    return new_dirty_rows, new_dirty_rows_labels, new_gt_indices


def cleanByIndex2(index_to_clean, clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices,
                  dirty_matrix):
    '''skip'''
    # # print('clean_rows')
    # # print(clean_rows)
    # # print('len(clean_rows)')
    # # print(len(clean_rows))
    # # print(type(clean_rows))
    # # print(type(clean_rows[1]))
    # # print(type(dirty_rows[index_to_clean][new_gt_indices[index_to_clean]]))
    # # print(type(clean_rows[-1]))
    # #clean_rows.append(dirty_rows[index_to_clean][new_gt_indices[index_to_clean]] if isinstance(dirty_rows[index_to_clean][new_gt_indices[index_to_clean]],list) else dirty_rows[index_to_clean][new_gt_indices[index_to_clean]].tolist())
    # clean_rows.append(dirty_rows[index_to_clean][new_gt_indices[index_to_clean]])
    # clean_rows_labels.append(dirty_rows_labels[index_to_clean])
    dirty_matrix = np.delete(dirty_matrix, index_to_clean, axis=0)
    new_dirty_rows = np.delete(dirty_rows, index_to_clean, axis=0)
    new_dirty_rows_labels = np.delete(dirty_rows_labels, index_to_clean)
    new_gt_indices = np.delete(new_gt_indices, index_to_clean)
    return new_dirty_rows, new_dirty_rows_labels, new_gt_indices, dirty_matrix


def replaceByMean(clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels):
    l = len(dirty_rows_labels)
    for i in range(l):
        clean_rows.append(dirty_rows[i][random.randint(0, 7)])
        clean_rows_labels.append(dirty_rows_labels[i])
    return clean_rows, clean_rows_labels


def replaceByGt(clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices):
    l = len(dirty_rows_labels)
    for i in range(l):
        clean_rows.append(dirty_rows[i][new_gt_indices[i]])
        clean_rows_labels.append(dirty_rows_labels[i])
    return clean_rows, clean_rows_labels


def discretize_feature(data, num_features):
    # 获取特征的最大最小值
    for feature_name in num_features:
        min_val = data[feature_name].min()
        max_val = data[feature_name].max()

        # 计算划分的间隔
        interval = (max_val - min_val) / 3

        # 进行离散化
        data[feature_name] = ((data[feature_name] - min_val) / interval).astype(int)

    return data
def calculate_SENS(output,y,len_clean,len_dirty):
    p=torch.abs(y-output)
    # print("p",p)
    # log_p=torch.log(p)
    sens1=torch.mean(torch.log(p[:len_clean]))
    # sens2= torch.mean((1 - p[len_clean+1:]) ** 2 * torch.log(p[len_clean+1:]))
    sens2= torch.mean( torch.log(p[len_clean+1:]))
    result=torch.abs(sens2-sens1)
    if torch.isnan(result):
       result = 1
    else:
        result = result.data
    if result>1:
        result=1
    return result
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Supreme')
parser.add_argument('--data_dir', default="/data/zjx/pycharm_project_841/data/data-reproduce")
parser.add_argument('--mv_type', default="systematic",
                        choices=["systematic", "random", "real"])
parser.add_argument('--input_size', default=7, type=int)
parser.add_argument('--val_size', default=50, type=int)
parser.add_argument('--data_size', default='2000', type=str)
parser.add_argument('--value', default=100, type=float)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--param_sens', default=0.1, type=float)
args = parser.parse_args()
load_dir = os.path.join(args.data_dir, args.data_size , args.dataset, args.mv_type)

# 训练模型
batch_size = 1000
val_size=args.val_size
# data, info = utils.load_space(os.path.join(args.space_dir, args.dataset))
data, info = utils.load_space(load_dir)
data = preprocess(data)  # 这么处理的原因是？
print('data after preprocess')
print(data)
# 候选集
X_train_repairs = np.array([data["X_train_repairs"][m] for m in data["repair_methods"]])
print(data["repair_methods"])
# print(data["repair_methods"])
X_train = pd.DataFrame(data["X_train_gt"])
Y_train = pd.DataFrame(data["y_train"])
X_val = pd.DataFrame(data["X_val"][:val_size])
Y_val = pd.DataFrame(data["y_val"][:val_size])
X_test = pd.DataFrame(data["X_test"])
Y_test = pd.DataFrame(data["y_test"])
num_features = X_train.columns
# X_test=discretize_feature(X_test,num_features)

# print(X_val)
# print(Y_val)
Y_train_label = LabelEncoder()
Y_train = Y_train_label.fit_transform(Y_train)
len_y_train=len(Y_train)
Y_val_label = LabelEncoder()
Y_val = Y_val_label.fit_transform(Y_val)
Y_test_label = LabelEncoder()
Y_test = Y_test_label.fit_transform(Y_test)
train_ds = tabularDataset(X_train, Y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

val_ds = tabularDataset(X_val, Y_val)
val_dl = DataLoader(val_ds, batch_size=val_size, shuffle=False)
test_ds = tabularDataset(X_test, Y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
x_data = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# x_data = [0.1,0.1,0.1,0.1]
# x_data = [0.5,0.6,0.7,0.8,0.9,1]

y_data = []
# e_list=[int(x*len()) for x in x_data]
# print(e_list)
MODEL = tabularModel(input_size=args.input_size)

# MODEL2 = tabularModel()
# 初始化
print("data['X_train_gt']")
print(data['X_train_gt'])
space, gt_indices, isTrue = make_space(X_train_repairs, X_val, gt=data['X_train_gt'])
# space_head=[['theta1', 'theta2', 'theta3', 'thetad1', 'thetad2', 'thetad3', 'tau1', 'tau2']]
# print('X_train_repairs')
# print(X_train_repairs)
print('gt_indices')
print(gt_indices)
print('isTrue')
print(isTrue)
indices = list(range(len(gt_indices)))
clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices,clean_indices,dirty_indices = select_clean(space, gt_indices,indices,isTrue,data['y_train'])

# 每一次循环，dirty_rows&label和clean_rows&label都要更新一次
# print('len(dirty_rows)', dirty_rows)
# print('dirty_rows_labels', dirty_rows_labels)
# dirty_rows, dirty_rows_labels,new_gt_indices=deleteByIndex(dirty_rows, dirty_rows_labels,temp_to_delete,new_gt_indices)
# np.random.seed(seed)  # numpy产生的随机数一致
# random.seed(seed)
# print('len(dirty_rows)2',dirty_rows)
# print('dirty_rows_labels', dirty_rows_labels)
sample_train_data, random_id_list = random_sample(dirty_rows)  # 行数，列数
print('len(sample_train_data)', len(sample_train_data))
# 设置一个和dirty_rows相同的weight矩阵，每个的权重, 0.099585062241, 1.0, 0.395833333333, 1.0, 0.098290598291, 0.0, 0.0, 1.0], [0.340659340659, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.191637630662, 1.0, 0.429696969697, 1.0, 0.099585062241, 1.0, 0.395833333333, 1.0, 0.098290598291, 0.0, 0.0, 1.0], [0.340659340659, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.191637630662, 1.0, 1.0, 1.0, 0.099585062241, 1.0, 0.395833333333, 1.0, 0.098290598291, 0.0, 0.0, 1.0], [0.340659340659, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.191637630662, 1.0, 1.0, 1.0, 0.099585062241, 1.0, 0.395833333333, 1.0, 0.098290598291, 0.0, 0.0, 1.0], [0.34065934065都设为1/M,即1/10

dirty_label = LabelEncoder()
y = dirty_label.fit_transform(dirty_rows_labels)
X = pd.DataFrame(sample_train_data)
# X=discretize_feature(X,num_features)
# 训练中设置的
dirty_ds = tabularDataset(X, y)
train_loader = DataLoader(dirty_ds, batch_size=batch_size, shuffle=False)

##################模型################################
# 学习率
LEARNING_RATE = 0.01
# BS
batch_size = 1000

# 训练前指定使用的设备
# DEVICE = torch.device('cuda')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数
criterion = nn.CrossEntropyLoss()
# 实例化模型
model = copy.deepcopy(MODEL)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



TOTAL_EPOCHS = 100
losses = []
len_dirty_rows = len(dirty_rows)
print('len_dirty_rows:', len_dirty_rows)
##################元数据（干净）##################################
############初始化权重############################
shape = np.shape(dirty_rows)
dirty_matrix = torch.zeros(shape[:2])
# print(dirty_matrix)
if len(clean_rows)==0:
    clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices = cleanByIndex(
        0, clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices)
    clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices = cleanByIndex(
        0, clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices)
if len(clean_rows)==1:
    clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices = cleanByIndex(
        0, clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices)
len_dirty_rows = len(dirty_rows)
print('len_dirty_rows:', len_dirty_rows)
model3 = copy.deepcopy(MODEL)
model3 = model3.to(DEVICE)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=LEARNING_RATE)
meta_label = LabelEncoder()
y = meta_label.fit_transform(clean_rows_labels)
X = pd.DataFrame(clean_rows)
# X=discretize_feature(X,num_features)
clean_ds = tabularDataset(X, y)
clean_dl = DataLoader(clean_ds, batch_size=batch_size)
print('clean_dl')

print(X)
model3.train()
# 要转换形式，不然会报错RuntimeError: grad can be implicitly created only for scalar outputs
criterion.reduction = 'mean'
# 训练100轮
TOTAL_EPOCHS = 100
# 记录损失函数
losses = []
# 干净数据集训练模型，用干净的验证集
for epoch in range(TOTAL_EPOCHS):
    for i, (x, y) in enumerate(clean_dl):
        x = x.float().to(DEVICE)  # 输入必须为float类型
        # print(type(x))
        # print(x)
        y = y.long().to(DEVICE)  # 结果标签必须为long类型
        # 清零
        optimizer3.zero_grad()
        outputs = model3(x)
        # 计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer3.step()
        losses.append(loss.cpu().data.item())
    print('Epoch : %d/%d,   Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS, np.mean(losses)))
model3.eval()
correct = 0
total = 0
# 验证
for i, (x, y) in enumerate(val_dl):
    # for i,(x, y) in enumerate(train_dl):
    x = x.float().to(DEVICE)
    y = y.long()
    # print(x)
    # print(y)
    outputs = model3(x).cpu()
    # print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    # print('------------------predicted------------------------')
    # print(predicted)
    # print('------------------i------------------------')
    # print(i)
    total += y.size(0)
    correct += (predicted == y).sum()
    # print(predicted.size())
    ############得到correct###################



start = time.time()
# sample选出来的训练集
# 训练模型，同时算参数，修改权重
SHOW_ACC = 0
n_arms=10
result_auc=[0 for x_d in range(len(x_data)) ]
algo= UCB1([], [],beta=args.value)
algo.initialize(n_arms)  # 初始化UCB，长度即为action个数,先设置len(arms)=10，即将脏数据集分成10份，当作10个arm
for ep in range(1, len_dirty_rows + 2):

    # np.random.seed(seed)  # numpy产生的随机数一致
    if ep == int(len_dirty_rows * x_data[SHOW_ACC]) + 1 or ep == len_dirty_rows + 1:

        SHOW_ACC = SHOW_ACC + 1
        # 用当前的清洗得到的干净数据集训练model2并用测试集测试model2 的精确度
        print("len(clean_rows)!!!!!!!!!!!!!!!!!!!!")
        print(len(clean_rows))
        sorted_clean_rows = [x for _, x in sorted(zip(clean_indices, clean_rows))]
        sorted_clean_rows_labels = [x for _, x in sorted(zip(clean_indices, clean_rows_labels))]

        # clean_rows,clean_rows_labels=replaceByMean(clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels)
        # clean_rows,clean_rows_labels=replaceByGt(clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels,new_gt_indices)
        # print(len(clean_rows))
        model2 = copy.deepcopy(MODEL)
        model2 = model2.to(DEVICE)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=LEARNING_RATE)
        meta_label = LabelEncoder()
        y = meta_label.fit_transform(sorted_clean_rows_labels)
        X = pd.DataFrame(sorted_clean_rows)
        # X = discretize_feature(X, num_features)
        clean_ds = tabularDataset(X, y)
        clean_dl = DataLoader(clean_ds, batch_size=batch_size)
        model2.train()
        # 要转换形式，不然会报错RuntimeError: grad can be implicitly created only for scalar outputs
        criterion.reduction = 'mean'
        # 训练100轮
        TOTAL_EPOCHS = 100
        # 记录损失函数
        losses = []
        # 干净数据集训练模型，用干净的验证集
        for epoch in range(TOTAL_EPOCHS):
            for i, (x, y) in enumerate(clean_dl):
                x = x.float().to(DEVICE)  # 输入必须为float类型
                # print(type(x))
                # print(x)
                y = y.long().to(DEVICE)  # 结果标签必须为long类型
                # 清零
                optimizer2.zero_grad()
                outputs = model2(x)
                # 计算损失函数
                loss = criterion(outputs, y)
                loss.backward()
                optimizer2.step()
                losses.append(loss.cpu().data.item())
            print('Epoch : %d/%d,   Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS, np.mean(losses)))
        model2.eval()
        correct = 0
        total = 0
        # 验证
        for i, (x, y) in enumerate(test_dl):
            # for i,(x, y) in enumerate(train_dl):

            # print(x)
            # print(y)
            x = x.float().to(DEVICE)
            y = y.long()
            # print(x)
            # print(y)
            outputs = model2(x).cpu()
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            # print('------------------predicted------------------------')
            # print(predicted)
            # print('------------------i------------------------')
            # print(i)
            total += y.size(0)
            correct += (predicted == y).sum()
            # print(predicted.size())
        print(correct)
        print(total)
        y_data.append(correct / total)
        print('准确率: %.4f %%' % (100 * correct / total))
        result_auc[SHOW_ACC - 1] = correct / total

        if ep == len_dirty_rows + 1:
            break
    # sample训练集
    sample_train_data, random_id_list = random_sample(dirty_rows)  # 行数，列数
    dirty_label = LabelEncoder()
    y = dirty_label.fit_transform(dirty_rows_labels)
    X = pd.DataFrame(sample_train_data)
    # X = discretize_feature(X, num_features)
    dirty_ds = tabularDataset(X, y)
    train_loader = DataLoader(dirty_ds, batch_size=batch_size, shuffle=False)
    # 验证集
    meta_label = LabelEncoder()
    y = meta_label.fit_transform(clean_rows_labels)
    X = pd.DataFrame(clean_rows)
    # X = discretize_feature(X, num_features)
    clean_ds = tabularDataset(X, y)
    meta_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)
    meta_loader = itertools.cycle(meta_loader)

    model.train()
    train_loss, train_acc = 0, 0

    # 跟之前不同，这里需要设置一个变量来记录weight（另外weight的更新方式，是覆盖还是累计？（（取平均）））
    w_list = torch.empty(0) #初始化tensor张量
    w_list = w_list.to(device=DEVICE)

    # 使用通过可信度，筛选好的train_loader训练
    # 在元学习之前计算一个元学习权重的归一化大小，
    dn = min(len(clean_rows * 10), len(dirty_rows))
    # print('dn', dn)
    # print("len(clean_rows)", len(clean_rows))
    tmp_criterion = nn.BCELoss()  # 二元交叉熵损失

    # y_w = torch.tensor(np.append(np.ones(len(clean_rows)), np.zeros(dn)))
    # y_tmp=torch.tensor
    # print('y_w', y_w)
    # print('sample_train_data',sample_train_data)
    random_indices = np.random.choice(np.array(sample_train_data).shape[0], size=dn, replace=False)
    random_rows = clean_rows[:]
    random_y=clean_rows_labels[:]
    for ri in random_indices:
        random_rows.append(sample_train_data[ri])
        random_y.append(dirty_rows_labels[ri])
    # random_rows=sample_train_data[random_indices]
    # print('random_rows',random_rows)
    # print('len_random_rows', len(random_rows))
    X_w = torch.tensor(random_rows)
    y_w=torch.tensor(random_y)
    y_w = y_w.to(DEVICE)
    # 前向传播
    X_w = X_w.to(torch.float32).to(DEVICE)
    # print("X_w.size()",X_w.size())
    outputs = model(X_w)
    # print(outputs)
    # print(y_w)
    # print("outputs.size()", outputs.size(), y_w.size())
    # # 计算NCE损失
    # outputs.to(torch.float32)
    SENS = calculate_SENS(outputs[:,1].to(torch.float32), y_w.to(torch.float32),len(clean_rows),dn)
    print("SENS",SENS)
    # w_loss = tmp_criterion(outputs[:, 0].unsqueeze(1).to(torch.float32), y_w.view(-1, 1).to(torch.float32))
    # print('w_loss', w_loss)
    # tmp_flag, tmp_sample = False, []

    # tmp_flag, tmp_sample = False, []
    for i, (inputs, labels) in enumerate(train_loader):  # 随机sample， batch_size=1024
        # print(i)
        if (inputs.size(0) <= 1):  #
            print('!!!!!!!!!!!!breakone!!!!!!!!!!!!!!!!')
            print('i:', i)
            # print(inputs.size())
            print(inputs.size(0))
            continue
        inputs, labels = inputs.float().to(device=DEVICE, non_blocking=True), \
            labels.long().to(device=DEVICE, non_blocking=True)
        optimizer.zero_grad()  # 将优化器中所有参数的梯度设置为零的函数
        with higher.innerloop_ctx(model, optimizer) as (meta_model, meta_opt):
            # 元数据是用清洗之后的数据代替。
            meta_train_outputs = meta_model(inputs)
            criterion.reduction = 'none'
            meta_train_loss = criterion(meta_train_outputs, meta_train_outputs)
            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=DEVICE)  # 创建元素值全为0的张量
            meta_train_loss = torch.sum(eps * meta_train_loss)
            meta_opt.step(meta_train_loss)

            # 2. Compute grads of eps on meta validation data（forward clean & backward clean）
            # 这里的metadata我们用训练集中干净的行来代替
            meta_inputs, meta_labels = next(meta_loader)
            meta_inputs, meta_labels = meta_inputs.float().to(device=DEVICE, non_blocking=True), \
                meta_labels.long().to(device=DEVICE, non_blocking=True)
            if (meta_inputs.size(0) <= 1):
                print(meta_inputs.size())
                print('!!!!!!!!!!!!breaktwo!!!!!!!!!!!!!!!!')
                continue
            meta_val_outputs = meta_model(meta_inputs)  # 谁的梯度大，谁的权重就大，对模型的影响就大
            # print(type(meta_val_outputs))
            # print(meta_val_outputs)
            criterion.reduction = 'mean'
            meta_val_loss = criterion(meta_val_outputs, meta_labels)
            eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

        # 3. Compute weights for current training batch
        # print('eps_grads')
        # print(eps_grads)#这个也会变成0
        w_tilde = -eps_grads # 用于对张量中的每个元素进行裁剪，将其限制在指定的范围内。
        l1_norm = torch.sum(w_tilde)
        if l1_norm != 0:
            w = w_tilde / l1_norm
        else:
            w =w_tilde  ####训练完之后记得修改权重，权重叠加

        # 4. Train model on weighted batch
        outputs = model(inputs)
        # print(outputs.size())
        criterion.reduction = 'none'
        minibatch_loss = criterion(outputs, labels)
        minibatch_loss = torch.sum(w * minibatch_loss)
        minibatch_loss.backward()
        optimizer.step()
        #############我们想同时利用这个w，来挑选需要清洗的行##########################
        w_list = torch.cat([w_list, w])  # 按序增加一节w_list
        s = args.param_sens
        if SENS == 0:
            SENS1 = 1.0 / s
        elif SENS == 1:
            SENS1 = 0
        else:
            SENS1 = s / SENS
        w_list=w_list*(SENS1)
        # w_list=w_list*(SENS)*(len_dirty_rows)*(1.0/len_y_train)
        # 每一轮挑选一行清洗，换成ground-truth
        # keep track of epoch loss/accuracy
        train_loss += minibatch_loss.item() * outputs.shape[0]
        _, pred_labels = torch.max(outputs.data, 1)
        train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
    dirty_row_len=len(w_list)
    for i in range(dirty_row_len):
        dirty_matrix[i, random_id_list[i]] = w_list[i]
    weight_sum_list = torch.sum(dirty_matrix, dim=1) 
    # sorted_idx = torch.argsort(weight_sum_list,stable=True, descending=True)
    sorted_arr = sorted(weight_sum_list)
    arr = [sorted_arr.index(x) for x in weight_sum_list]#idx:大小weight_sum_list标识
    indexed_arr = [(i, x) for i, x in enumerate(arr)]
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1])
    # print(sorted_arr)
    idx = [x[0] for x in sorted_arr]
    if len(idx) == 0:
        continue
    chosen_arm = algo.select_arm()  
    print('chosen_arm')
    print(chosen_arm)
    index_to_clean=idx[int(((chosen_arm+1)/n_arms)*(dirty_row_len-1))]
    # index_to_clean=index_to_clean.item()
    #index_to_clean = torch.argmin(weight_sum_list)  # 权重之和最小的最需要清洗
    print('index_to_clean')
    print(index_to_clean)

    # if ep>int(len_dirty_rows*0.4) and ep<=int(len_dirty_rows*0.5):
    #     dirty_rows, dirty_rows_labels, new_gt_indices,dirty_matrix=cleanByIndex2(index_to_clean, clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices,dirty_matrix)
    # else:
    #     clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices,dirty_matrix=cleanByIndex(index_to_clean, clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices,dirty_matrix)
    clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices = cleanByIndex(
        index_to_clean, clean_rows, clean_rows_labels, dirty_rows, dirty_rows_labels, new_gt_indices, dirty_matrix,clean_indices,dirty_indices)

    # 输出每个元素在排序后的数组中的大小顺序和索引
    # print(idx)

    model3 = copy.deepcopy(MODEL)
    model3 = model3.to(DEVICE)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=LEARNING_RATE)
    meta_label = LabelEncoder()
    y = meta_label.fit_transform(clean_rows_labels)
    X = pd.DataFrame(clean_rows)
    # X = discretize_feature(X, num_features)
    clean_ds = tabularDataset(X, y)
    clean_dl = DataLoader(clean_ds, batch_size=batch_size)
    model3.train()
    # 要转换形式，不然会报错RuntimeError: grad can be implicitly created only for scalar outputs
    criterion.reduction = 'mean'
    TOTAL_EPOCHS = 100
    losses = []
    for epoch in range(TOTAL_EPOCHS):
        for i, (x, y) in enumerate(clean_dl):
            if (x.size(0) <= 1):  #
                print('!!!!!!!!!!!!breakone!!!!!!!!!!!!!!!!')
                print('i:', i)
                # print(inputs.size())
                print(x.size(0))
                continue
            x = x.float().to(DEVICE)  
            # print(type(x))
            # print(x)
            y = y.long().to(DEVICE) 
            optimizer3.zero_grad()
            outputs = model3(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer3.step()
            losses.append(loss.cpu().data.item())
        # print('Epoch : %d/%d,   Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS, np.mean(losses)))
    model3.eval()
    new_correct = 0
    total = 0
    # 验证
    for i, (x, y) in enumerate(val_dl):
        # for i,(x, y) in enumerate(train_dl):

        # print(x)
        # print(y)
        x = x.float().to(DEVICE)
        y = y.long()
        # print(x)
        # print(y)
        outputs = model3(x).cpu()
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print('------------------predicted------------------------')
        # print(predicted)
        # print('------------------i------------------------')
        # print(i)
        total += y.size(0)
        new_correct += (predicted == y).sum()
        # print(predicted.size())
    reward=(new_correct-correct)/correct
    # print('reward')
    # print(reward)
    correct=new_correct
    algo.update(chosen_arm, reward)
result=0
end = time.time()
running_time = end - start
print('time cost : %.5f sec' % running_time)
for x_d in range(len(x_data)-1):
    result+=(result_auc[x_d]+result_auc[x_d+1])*0.1/2
print(result)




