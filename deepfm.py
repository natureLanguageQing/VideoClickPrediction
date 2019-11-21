"""
模型：DeepFM

运行环境： DeepCTR-Torch (https://github.com/shenweichen/DeepCTR-Torch)


特征说明

1.用户特征
用户原始特征：gender、frequency、A1、...
用户关注和感兴趣的topics数目

2.问题特征
问题标题的字、词计数
问题描述的字、词计数
问题绑定的topic数目

3.用户问题交叉特征
用户关注、感兴趣的话题和问题绑定的话题交集计数
邀请距离问题创建的天数

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from utils import *

""" 运行DeepFM """

path = 'data/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# 测试
# train = train[0:10000]
# test = test[0:10000]

# print(train.head())
data = pd.concat([train, test], ignore_index=True, sort=False)
# print(data.head())
# timestamp：代表改用户点击改视频的时间戳，如果未点击则为NULL。 deviceid：用户的设备id。 newsid：视频的id。 guid：用户的注册id。 pos：视频推荐位置。 app_version：app版本。 device_vendor：设备厂商。 netmodel：网络类型。 osversion：操作系统版本。 lng：经度。 lat：维度。 device_version：设备版本。 ts：视频暴光给用户的时间戳。
# id,target,timestamp,deviceid,newsid,guid,pos,app_version,device_vendor,netmodel,osversion,lng,lat,device_version,ts
# id,deviceid,newsid,guid,pos,app_version,device_vendor,netmodel,osversion,lng,lat,device_version,ts
# 单值类别特征
fixlen_category_columns = ['app_version', 'device_vendor', 'netmodel', 'osversion', 'device_version']
# 数值特征
fixlen_number_columns = ['timestamp', 'lng', 'lat', 'ts']

target = ['target']

data[fixlen_category_columns] = data[fixlen_category_columns].fillna('-1', )
data[fixlen_number_columns] = data[fixlen_number_columns].fillna(0, )

for feat in fixlen_category_columns:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data[fixlen_number_columns] = mms.fit_transform(data[fixlen_number_columns])

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in fixlen_category_columns] + [DenseFeat(feat, 1, ) for feat in
                                                                  fixlen_number_columns]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train = data[~data['target'].isnull()]
test = data[data['target'].isnull()]

train, vaild = train_test_split(train, test_size=0.2)
train_model_input = {name: train[name] for name in feature_names}
vaild_model_input = {name: vaild[name] for name in feature_names}

device = 'cuda:0'
"""第一步：初始化一个模型类"""
model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary',
               l2_reg_embedding=1e-5, device=device)

"""第二步：调用compile()函数配置模型的优化器、损失函数、评价函数"""
model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )

"""第三步：调用fit()函数训练模型"""
model.fit(train_model_input, train[target].values, batch_size=8192, epochs=10,
          validation_data=[vaild_model_input, vaild[target].values], verbose=1,
          model_cache_path='models\\deepfm.model')

"""预测"""
test_model_input = {name: test[name] for name in feature_names}
pred_ans = model.predict(test_model_input, 8192)
pred_ans = pred_ans.reshape(pred_ans.shape[0])
# result = test['id']
# result.loc[:, 'result'] = pred_ans
with open("result.csv", "w") as f:
    f.write("id,target\n")
    for i, j in zip(test['id'].values.tolist(), pred_ans):
        f.write(str(i) + "," + str(j) + "\n")
