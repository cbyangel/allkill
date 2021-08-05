
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import requests
import json
import pandas as pd
import datetime
import time
import sys
import numpy as np
import logging
import math

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from datetime import date, timedelta
from redis._compat import xrange
from rediscluster import StrictRedisCluster
from collections import defaultdict

from pydruid.client import *
from pylab import plt
from pydruid.utils.aggregators import *
from pydruid.utils.filters import *
from pydruid.utils.dimensions import *

from copy import deepcopy
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Input, Dense, Embedding, Flatten, Dropout, merge, Activation, BatchNormalization, LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import add, concatenate
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras.utils import plot_model


import warnings
warnings.filterwarnings('ignore')


from jarvis_common.hive import C_HIVE
hive_cmd = C_HIVE(user='b_dataengineering') 



class FeatureSelector():
    def __init__(self, feature_names):
        self._feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[ self._feature_names ]
    
    
class ItemMapper():
    def __init__(self, allkill):
        self._allkill = allkill
        
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        itemID_dict = {}
        
        for idx, item in enumerate(set(X['itemID'].unique().tolist() + self._allkill)):
            itemID_dict[item] = idx
        
        return itemID_dict
    
    
class UserMapper():
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        userID_dict = {}
        
        for idx, user in enumerate(X.sort_values(by=['userID'], ascending=True)['userID'].unique()):
            userID_dict[user] = idx
        
        return userID_dict
    
    
    
class DeepAePreProcessor():
    
    def __init__(self, userID_dict, itemID_dict):
        self._userID_dict = userID_dict
        self._itemID_dict = itemID_dict
        
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        df = deepcopy(X)
        df['userID'] = df['userID'].apply(lambda x: self._userID_dict[x])
        df['itemID'] = df['itemID'].apply(lambda x: self._itemID_dict[x])
        return self.filterItemID(df)   
        
    
    def filterItemID(self, X, y=None):
        element_group_sizes = X['itemID'].groupby(X['itemID']).transform('size')
        target_list = X[element_group_sizes < 3]['itemID'].tolist()
        X['isTarget'] = X['itemID'].apply(lambda x: True if x in target_list else False)
        X = X[X['isTarget'] == False]
        X = X[['userID', 'itemID', 'rating']]
        return X
    
    

    
class DeepAeTransformer():
    
    def __init__(self, userID_dict, itemID_dict):
        self._userID_dict = userID_dict
        self._itemID_dict = itemID_dict
        
        
    def fit(self, X, y=None):
        return X
        
    
    def transform(self, X, y=None):
        num_users = len(self._userID_dict.keys())
        num_items = len(self._itemID_dict.keys())
        #print(num_users, num_items)
        
        return self.getMatrix(X, num_items, num_users)
        
    
    def getMatrix(self, X, num_items, num_users, init_value=0.0):
        matrix = np.full((num_items, num_users), init_value)
        for (_, userID, itemID, rating) in X.itertuples():
            matrix[itemID, userID] = rating
        return matrix
    
    
    
class CategoricalTransformer():
    
    def __init__(self, itemID_dict):
        self._itemID_dict = itemID_dict
        
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        repeat_dict = dict(zip(X.itemno, X.repeat_code))
        
        feat_dict = {'itemID':[], 'repeat_code':[]}
        for key in self._itemID_dict.keys():
            feat_dict['itemID'].append(key)
            feat_dict['repeat_code'].append(repeat_dict[key] if key in repeat_dict else 1)
           
        feat_df = pd.DataFrame(feat_dict)
        feat_df['repeat_code'] = preprocessing.LabelEncoder().fit(feat_df['repeat_code']).transform(feat_df['repeat_code'])
        onehot_df = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False).fit(feat_df[['repeat_code']]).transform(feat_df[['repeat_code']])
        return onehot_df
    
    

    
class Deep_AE_Model():
    def __init__(self, layers, activation, last_activation, dropout, regularizer_encode, regularizer_decode, side_infor_size ):
        self._layers = layers
        self._activation = activation
        self._last_activation = last_activation
        self._dropout = dropout
        self._regularizer_encode = regularizer_encode
        self._regularizer_decode = regularizer_decode
        self._side_infor_size = side_infor_size
        
    
    def build(self, X):
        '''
        Build Deep AE for CF
            INPUT: 
                X: #_user X #_item matrix
                layers: List, each element is the number of neuron for a layer
                reg: L2 regularization parameter
                activation: activation function for all dense layer except the last
                last_activation: activation function for the last dense layer
                dropout: dropout rate
                regularizer_encode: regularizer for encoder
                regularizer_decode: regularizer for decoder
                side_infor_size: size of the one hot encoding side information
            OUTPUT:
                Keras model
        '''

        # Input
        input_layer = x = Input(shape=(X.shape[1],), name='UserRating')

        # Encoder
        # -----------------------------
        k = int(len(self._layers)/2)
        i = 0
        for l in self._layers[:k]:
          x = Dense(l, activation=self._activation,
                          name='EncLayer{}'.format(i), kernel_regularizer=regularizers.l2(self._regularizer_encode))(x)
          i = i+1


        # Latent Space
        # -----------------------------
        x = Dense(self._layers[k], activation=self._activation, 
                                    name='LatentSpace', kernel_regularizer=regularizers.l2(self._regularizer_encode))(x)

        # Dropout
        x = Dropout(rate = self._dropout)(x)

        # Decoder
        # -----------------------------
        for l in self._layers[k+1:]:
          i = i-1
          x = Dense(l, activation=self._activation, 
                          name='DecLayer{}'.format(i), kernel_regularizer=regularizers.l2(self._regularizer_decode))(x)

        # Output

        output_layer = Dense(X.shape[1]-self._side_infor_size, activation=self._last_activation,    
                             name='UserScorePred',kernel_regularizer=regularizers.l2(self._regularizer_decode))(x)

        # this model maps an input to its reconstruction
        model = Model(input_layer, output_layer)
        return model
    

################################################################################################################   
    

def getAllkillList():
        api_url = "http://searchapi.auction.co.kr/api/search/AllKill/GetAllKillRecmTabDataView"
        data = requests.get(api_url).json()
        allkillitem = []
        allkillManageItem = []
        for allkillDic in data["Data"]:
            if allkillDic["Item"] != None:
                allkillitem.append(allkillDic["Item"]["ID"].lower())
                if int(allkillDic["Item"]["VisibleOrderby"]) < 9999:
                    allkillManageItem.append(allkillDic["Item"]["ID"].lower())

        return allkillitem, allkillManageItem
    


def getOrderData():
    now = datetime.datetime.now()
    b1now = now + datetime.timedelta(days=-7)

    strSdt = b1now.strftime('%Y-%m-%dT%H:%M:%S+09:00')
    strEdt = now.strftime('%Y-%m-%dT%H:%M:%S+09:00')

    query = PyDruid('http://druid-broker.gmarket.co.kr', 'druid/v2')
    ext_fn = RegexExtraction(r'(\w\w).*')

    group = query.groupby(
        datasource='ub_page_event_v2_iac',
        granularity='all',
        intervals=strSdt+'/'+strEdt,
        dimensions=["goodscode"
                    ,"hour"
                    ,DimensionSpec('cluster_no', 'cluster_no', extraction_function=ext_fn).build()],
        filter=(Dimension("area_code") == "op") & ## 주문
               (Dimension("from_area_code") == "100000315") ,
        #filter=(Dimension("area_code") == "op"),
        aggregations={"count": doublesum("count"),"order_amnt": doublesum("order_amnt"), "distinct_cguid" : cardinality("cguid")}
    )
    df = query.export_pandas()
    df.distinct_cguid = df.distinct_cguid.round(0)

    dfrd = df.groupby(['goodscode','hour','cluster_no']).sum().reset_index().copy()

    gdfrd = dfrd.groupby(['cluster_no','hour']).max().reset_index()[['cluster_no','hour','distinct_cguid','order_amnt']]
    gdfrd = gdfrd.rename(columns={"distinct_cguid": "max_cguid", "order_amnt":"max_amnt"})

    dfrd = dfrd.merge(gdfrd,on=['cluster_no','hour'])

    dfrd['userID']=dfrd.cluster_no + dfrd.hour

    dfrd = dfrd.rename(columns={'goodscode':'itemID', 'distinct_cguid':'rating'})
    dfrd['rating'] = dfrd['rating'].astype(int)
    dfrd = dfrd[['userID', 'itemID', 'rating']]

    return dfrd, gdfrd



def getDateVar(dur=5):
    # 학습 데이터 기간 선정
    import datetime
    now = datetime.datetime.now()
    rangeDate = ''
    dateList = []
    for d in range(dur):
        chgdt = now + datetime.timedelta(days=-d)
        rangeDate += chgdt.strftime('%Y%m%d') + ','
        dateList.append(chgdt.strftime('%Y%m%d'))
    rangeDate = rangeDate[:-1]
    processDt = dateList[0]
    return rangeDate, dateList



def getOrderDataHive():
    rangeDate, dateList = getDateVar(6)
    q  = '''
    set tez.queue.name=business-low;
    select c.itemno goodscode, substring(o.ins_date,12,2) hour, cluster_no cluster_no, count(*) count, sum(order_amount) order_amnt, count(distinct o.cguid) distinct_cguid
    from baikal_analytics.ub_fltr_frm_cart_iac c
    join baikal_analytics.ub_fltr_frm_order_iac o
        on c.dt = o.dt
        and c.cartno = o.cartno
    join baikal_analytics.allkill_cluster_result clu
        on o.cguid = clu.cguid
        and clu.dt = $cdt
    where c.dt in ($dt) 
    and c.area_code in (100000315, 100000324,100001107,100001111,100000502)
    group by c.itemno, substring(o.ins_date,12,2), cluster_no
    '''
    q = q.replace('$dt', rangeDate)
    q = q.replace('$cdt', '20210701')
    # print(q)
    rd = hive_cmd.post(q)
    
    df = pd.DataFrame(rd['results'], columns=['goodscode','hour','cluster_no','count','order_amnt','distinct_cguid'])
    df.loc[df.cluster_no.str.len()<2, 'cluster_no'] = '0' + df.loc[df.cluster_no.str.len()<2, 'cluster_no']
    dfrd = df.groupby(['goodscode','hour','cluster_no']).sum().reset_index().copy()
    gdfrd = dfrd.groupby(['cluster_no','hour']).max().reset_index()[['cluster_no','hour','distinct_cguid','order_amnt']]
    gdfrd = gdfrd.rename(columns={"distinct_cguid": "max_cguid", "order_amnt":"max_amnt"})
    dfrd = dfrd.merge(gdfrd,on=['cluster_no','hour'])
    dfrd['userID']=dfrd.cluster_no + dfrd.hour
    dfrd = dfrd.rename(columns={'goodscode':'itemID', 'distinct_cguid':'rating'})
    dfrd = dfrd[['userID', 'itemID', 'rating']]
    return dfrd, gdfrd


def getRepeatCd(path):
    q = "select itemno, repeat_cguid_cnt from baikal_analytics.boycho_iac_item_features"
    df = hive_cmd.get(q)['results']
    
    p90, p95, p99 = df[df['repeat_cguid_cnt'] > 0]['repeat_cguid_cnt'].describe(percentiles = [.9, .95, .99]).loc[['90%', '95%', '99%']].values
    df['repeat_code'] = df['repeat_cguid_cnt'].apply(lambda x: 1 if x < p90 else 2 if ((x >= p90) and (x < p95)) else 3 if ((x >= p95) and (x < p99)) else 4)
   
    df[['itemno', 'repeat_code']].to_csv(path, header=False, index=False)
    

def rmvFile(path):
    if os.path.isfile(path):
        os.remove(path)
        

def masked_se(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1)
    return masked_mse




def masked_mse(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
    return masked_mse



def masked_rmse(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_mse


def masked_rmse_clip(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred = K.clip(y_pred, 1, 5)
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_mse


def set_AIRedis(_dfMerge, allkillManageItem):
        
        now = datetime.datetime.now()
        tom = now + datetime.timedelta(days=1)
        timediff = datetime.datetime(tom.year, tom.month, tom.day)
        td = timediff - now
        exp_sec = int(td.total_seconds())
        p_date = now.strftime('%Y%m%d')
        
        #set ai2 server
        startup_nodes = [
           {"host":"172.30.206.158", "port":"7000"}
          ,{"host":"172.30.206.159", "port":"7000"}
          ,{"host":"172.30.206.160", "port":"7000"}
          ,{"host":"172.30.206.161", "port":"7000"}
          ,{"host":"172.30.206.162", "port":"7000"}
          ,{"host":"172.30.206.163", "port":"7000"}]
        rc = StrictRedisCluster(startup_nodes=startup_nodes, max_connections=32, decode_responses=True)

        ## 전략구좌가 개수 확인
        isAdjustPriorityCluster = True
#         isAdjustPriorityCluster = False
#         if len(allkillManageItem)> 25:
#             isAdjustPriorityCluster = True

        for x in set(_dfMerge.userID):
            clusterno = x[:2]
            if (isAdjustPriorityCluster == True):
                ## 빅스마일기간동안 전략구좌가 있으면 전략구좌를 우선한다.
                _dfMerge = _dfMerge[~_dfMerge.itemID.isin(allkillManageItem)]
                v = allkillManageItem + list(_dfMerge[_dfMerge.userID == x ].sort_values(['rnk'])['itemID'])
            else:
                v = list(_dfMerge[_dfMerge.userID == x ].sort_values(['rnk'])['itemID'])


            for str_z in ['00']:

                new_key = 'iac:personalization:deal:date:' + p_date + ':cluster:' + clusterno + str_z + ':itemnos'
#                 print(new_key, len(v))
                rc.delete(new_key)
                rc.rpush(new_key, *v)
                rc.expire(new_key,exp_sec)



################################################################################################################   


start = time.time()

dt_1day_ago = (datetime.datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
file_1day_ago = ".".join(["_".join(['Features', dt_1day_ago]), 'csv'])

dt = datetime.datetime.now().strftime('%Y%m%d')
file_now = ".".join(["_".join(['Features', dt]), 'csv'])
if os.path.exists(file_now) == False:
    getRepeatCd(file_now)
else:
    rmvFile(file_1day_ago)
    repeat_df = pd.read_csv(file_now, sep=',', names=['itemno', 'repeat_code'])
    allkill_list, allkillManageItem = getAllkillList()
    dfrd, gdfrd = getOrderData()

    features = ['userID', 'itemID', 'rating']

    u_mapper_steps = [
        ('feat_selector', FeatureSelector(features) ),
        ('UserMapper', UserMapper())
    ]
    pipeline_list = [("UserMapper", Pipeline(u_mapper_steps))]
    u_mapper_pipeline = FeatureUnion(transformer_list=pipeline_list)
    u_mapper_pipeline.fit(dfrd)
    userID_dict = u_mapper_pipeline.transform(dfrd)[0]


    i_mapper_steps = [
        ('feat_selector', FeatureSelector(features) ),
        ('ItemMapper', ItemMapper(allkill_list))
    ]
    pipeline_list = [("ItemMapper", Pipeline(i_mapper_steps))]
    i_mapper_pipeline = FeatureUnion(transformer_list=pipeline_list)
    i_mapper_pipeline.fit(dfrd)
    itemID_dict = i_mapper_pipeline.transform(dfrd)[0]



    preprocessor_steps = [
        ('DeepAePreProcessor', DeepAePreProcessor(userID_dict, itemID_dict))
    ]
    pipeline_list = [("DeepAePreProcessor", Pipeline(preprocessor_steps))]
    pre_processor_pipeline = FeatureUnion(transformer_list=pipeline_list)
    pre_processor_pipeline.fit(dfrd)
    df = pd.DataFrame(pre_processor_pipeline.transform(dfrd), columns=features)



    train_df, test_df = train_test_split(df, 
                                         stratify=df['itemID'],
                                         test_size=0.1,
                                         random_state=999613182)

    train_df, validate_df = train_test_split(train_df,
                                     stratify=train_df['itemID'],
                                     test_size=0.1,
                                     random_state=999613182)



    transformer_steps = [
        ('DeepAeTransformer', DeepAeTransformer(userID_dict, itemID_dict))
    ]
    transformer_pipeline_list = [("DeepAeTransformer", Pipeline(transformer_steps))]
    transformer_pipeline = FeatureUnion(transformer_list=transformer_pipeline_list)
    transformer_pipeline.fit(train_df)
    users_items_matrix_train_zero = transformer_pipeline.transform(train_df)
    transformer_pipeline.fit(test_df)
    users_items_matrix_test = transformer_pipeline.transform(test_df)
    transformer_pipeline.fit(validate_df)
    users_items_matrix_validate = transformer_pipeline.transform(validate_df)


    transformer_steps = [
        ('CategoricalTransformer', CategoricalTransformer(itemID_dict))
    ]
    transformer_pipeline_list = [("CategoricalTransformer", Pipeline(transformer_steps))]
    transformer_pipeline = FeatureUnion(transformer_list=transformer_pipeline_list)
    transformer_pipeline.fit(repeat_df)
    onehot_df = transformer_pipeline.transform(repeat_df)


    user_items_user_info = np.concatenate((users_items_matrix_train_zero, onehot_df), axis=1)



    Deep_AE = Deep_AE_Model([128, 64, 32, 64, 128], 'selu', 'selu', 0.5, 0.001, 0.001, 4)
    Deep_AE = Deep_AE.build(user_items_user_info)
    Deep_AE.compile(optimizer = Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip]) 
    #Deep_AE.summary()


    hist_Deep_AE = Deep_AE.fit(x=user_items_user_info, y=users_items_matrix_train_zero,
                      epochs=300,
                      batch_size=128,
                      validation_data=(user_items_user_info, users_items_matrix_validate), verbose=0) # 2 all output


    predict_deep = Deep_AE.predict(user_items_user_info)
    test_result_deep = Deep_AE.evaluate(user_items_user_info, users_items_matrix_test, verbose=0)
    loss = K.eval(masked_rmse_clip( 
        K.constant((users_items_matrix_train_zero)), 
        K.constant(predict_deep)))
    print(np.mean(loss))


    num_users = len(userID_dict.keys())
    num_items = len(itemID_dict.keys())
    print(num_users, num_items)
    result_dict = {'userID':[], 'itemID':[], 'score':[]}
    result_df = None
    for i in itemID_dict.keys():
        item_list = [ i for u in range(num_users)]
        user_list = list(userID_dict.keys())
        score_list = predict_deep[itemID_dict[i]].tolist()
        result_dict = {'userID':user_list, 'itemID':item_list, 'score':score_list}
        result_df = pd.concat([result_df, pd.DataFrame(result_dict)])


    n_hour = datetime.datetime.now().strftime('%H')
    result_df = result_df[(result_df['userID'].str[2:] == n_hour)]
    result_df = result_df[['userID', 'itemID', 'score']]
    result_df['isChk'] = result_df['itemID'].apply(lambda x :  1 if x in allkill_list else 0 )
    result_df = result_df[result_df.isChk == 1][['userID', 'itemID', 'score']]
    result_df['rnk'] = result_df.groupby('userID')['score'].rank(method='first', ascending=False)
    result_df = result_df.sort_values(by=['userID', 'score'], ascending=['True', 'False'])

    set_AIRedis(result_df, allkillManageItem)

print("time :", time.time() - start)
