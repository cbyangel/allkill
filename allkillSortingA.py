import os
import requests
import json
import pandas as pd
import datetime
import time
import sys
import numpy as np
import logging
import random

from redis._compat import xrange
from rediscluster import StrictRedisCluster
from collections import defaultdict

from pydruid.client import *
from pylab import plt
from pydruid.utils.aggregators import *
from pydruid.utils.filters import *
from pydruid.utils.dimensions import *

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k
from lightfm.cross_validation import random_train_test_split

from jarvis_common.hive import C_HIVE



class BaseDB(object):
    def __init__(self):
        self._hive_cmd = C_HIVE(user='b_dataengineering')
        self._query = PyDruid('http://druid-broker.gmarket.co.kr', 'druid/v2')
        self._ext_fn = RegexExtraction(r'(\w\w).*')
        
    def getHiveData(self, query):
        df = self._hive_cmd.get(query)['results']
        return df
    
class Behaviors(BaseDB):
    def __init__(self):
        BaseDB.__init__(self)
        self._df = None
        self._ulist = None
        self._ilist = None
        self._udict = None
        self._idict = None
         
    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, df):
        if df is None:
            return ValueError('interaction is empty')
        self._df = df
        
            
    @property
    def ulist(self):
        return self._ulist
    
    @ulist.setter
    def ulist(self, ulist):
        if ulist is None:
            return ValueError('uid_list is empty')
        self._ulist = ulist    
        
        
    @property
    def udict(self):
        return self._udict
    
    @udict.setter
    def udict(self, udict):
        if udict is None:
            return ValueError('uid_dict is empty')
        self._udict = udict    
        
        
    @property
    def ilist(self):
        return self._ilist
    
    @ilist.setter
    def ilist(self, ilist):
        if ilist is None:
            return ValueError('iid_list is empty')
        self._ilist = ilist    
        
        
    @property
    def idict(self):
        return self._idict
    
    @idict.setter
    def idict(self, idict):
        if idict is None:
            return ValueError('iid_dict is empty')
        self._idict = idict
               
        
    def getDruidOrd(self):
        now = datetime.datetime.now()
        b1now = now + datetime.timedelta(days=-7)

        strSdt = b1now.strftime('%Y-%m-%dT%H:%M:%S+09:00')
        strEdt = now.strftime('%Y-%m-%dT%H:%M:%S+09:00')
        
        group = self._query.groupby(
            datasource='ub_page_event_v2_iac',
            granularity='all',
            intervals=strSdt+'/'+strEdt,
            dimensions=["goodscode"
                        ,"hour"
                        ,DimensionSpec('cluster_no', 'cluster_no', extraction_function=self._ext_fn).build()],
            filter=(Dimension("area_code") == "op") & ## 주문
                   (Dimension("from_area_code") == "100000315") ,
            aggregations={"count": doublesum("count"),"order_amnt": doublesum("order_amnt"), "distinct_cguid" : cardinality("cguid")}
        )
        df = self._query.export_pandas()
        df.distinct_cguid = df.distinct_cguid.round(0)

        dfrd = df.groupby(['goodscode','hour','cluster_no']).sum().reset_index().copy()

        gdfrd = dfrd.groupby(['cluster_no','hour']).max().reset_index()[['cluster_no','hour','distinct_cguid','order_amnt']]
        gdfrd = gdfrd.rename(columns={"distinct_cguid": "max_cguid", "order_amnt":"max_amnt"})

        dfrd = dfrd.merge(gdfrd,on=['cluster_no','hour'])

        dfrd['rating'] = np.round(dfrd.distinct_cguid / dfrd.max_cguid, 2)
        dfrd['ratingAmnt'] = np.round(dfrd.order_amnt / dfrd.max_amnt, 2)

        dfrd['userID']=dfrd.cluster_no + dfrd.hour
        dfrd = dfrd.rename(columns={'goodscode':'itemID'})
        self._df = dfrd[['userID', 'itemID', 'rating']]
        
    
    
    def setFeatMapping(self, allkill_list):
        uid_dict = defaultdict()
        iid_dict = defaultdict()
        
        uid_list = self._df['userID'].unique().tolist()
        iid_list = self._df['itemID'].unique().tolist()
        iid_list = iid_list + list(set(allkill_list) - set(iid_list))

        for i, v in enumerate(uid_list):
            uid_dict.setdefault(v, i)

        for i, v in enumerate(iid_list):
            iid_dict.setdefault(v, i)

        self._ulist = uid_list
        self._udict = uid_dict
        self._ilist = iid_list
        self._idict = iid_dict
        
        
        
class Features(BaseDB):
    def __init__(self, features):
        BaseDB.__init__(self)
        self.features = features
        self._idf = None
        self._ifeatures = []
        
    @property
    def idf(self):
        return self._idf
    
    @idf.setter
    def idf(self, idf):
        if idf is None:
            return ValueError('item_df is empty')
        self._idf = idf
        
            
    @property
    def ifeatures(self):
        return self._ifeatures
    
    @ifeatures.setter
    def ifeatures(self, ifeatures):
        if len(ifeatures) == 0:
            return ValueError('ifeatures is empty')
        self._ifeatures = ifeatures
        
    
    def getSqlOrd(self, iid_list, isNumeric=True):
        query = None
        item_list = "'" + "','".join(item.replace("'", r"\'") for item in iid_list)+ "'" 
        if isNumeric:
            query = "select * from baikal_analytics.boycho_iac_item_features where itemno in ({0})".format(item_list)
        else:
            query = "select * from baikal_analytics.boycho_iac_item_ekws_info where itemno in ({0})".format(item_list)
        self._idf = self._hive_cmd.get(query)['results']
        
        
    def make_item_features(self, isNumeric=True):
        #item_list = []
        try:
            item_idx = len(self.features) - 1
            for x in self._idf[self.features].values:
                if isNumeric:
                    dict_ = { f : v for f, v in zip(self.features, x) if f != 'itemno' }
                    self._ifeatures.append((x[item_idx], dict_))
                else:
                    dict_ = [ v for f, v in zip(self.features, x) if f != 'itemno' ]
                    if dict_[0] != '':
                        self._ifeatures.append((x[item_idx], dict_))
        except Exception as e:
             print(str(e))

                
class lightfm(Behaviors, Features):
    def __init__(self, features, num_threads: int = 2, num_components: int = 30, num_epochs: int = 5, item_alpha: float= 1e-6):
        Behaviors.__init__(self)
        Features.__init__(self, features)
        self.num_threads = num_threads
        self.num_components = num_components
        self.num_epochs = num_epochs
        self.item_alpha = item_alpha
        self.dataset1 = Dataset()
        self.interactions = None
        self.weights = None
        self.item_features = None
        self.model = None
    
        
    def __len__(self):
        return len(self.ulist)
    
        
    def make_dataset(self, isNumeric=True):
       
        self.dataset1.fit(
            self.ulist,
            self.ilist,
            item_features = self.features if isNumeric else self.idf[self.features[0]].unique()
        )
        self.interactions, self.weights = self.dataset1.build_interactions( [ (x[0], x[1], x[2]) for x in self.df[['userID', 'itemID', 'rating' ]].values  ])
        self.item_features = self.dataset1.build_item_features(self.ifeatures, normalize=isNumeric)     
 
       

    
    def split_train_test(self, test_percent=0.2):
        train, test = random_train_test_split(self.interactions, test_percentage=test_percent, random_state=np.random.RandomState(3))
        train_weights, test_weights = random_train_test_split(self.weights, test_percentage=test_percent, random_state=np.random.RandomState(3))
        return train, test
    
    
    
    def make_model(self, train):
        # Let's fit a WARP model: these generally have the best performance.
        model = LightFM(loss='warp',
                item_alpha=self.item_alpha,
                no_components=self.num_components)

        # Run 3 epochs and time it.
        model = model.fit(train, item_features=self.item_features, epochs=self.num_epochs, num_threads=self.num_threads)
        return model
    
        
    def calc_auc(self, model, train, test):
        # Compute and print the AUC score
        train_auc = auc_score(model, train, item_features=self.item_features, num_threads=self.num_threads).mean()
        print('Collaborative filtering train AUC: %s' % train_auc)

        # We pass in the train interactions to exclude them from predictions.
        # This is to simulate a recommender system where we do not
        # re-recommend things the user has already interacted with in the train
        # set.
        test_auc = auc_score(model, test, train_interactions=train, item_features=self.item_features, num_threads=self.num_threads).mean()
        print('Collaborative filtering test AUC: %s' % test_auc)
     
    
    def predict(self, model, allkill_list):
       
        all_n = defaultdict(list)
        all_scores = np.empty(shape=(0,len(self.ilist)))
     
        for u in self.ulist:
            scores = model.predict(self.udict[u], np.arange(len(self.ilist)), item_features=self.item_features)
            all_scores = np.vstack((all_scores, scores))
            
        for u in self.ulist:
            for i in self.ilist:
                all_n[u].append((i, all_scores[self.udict[u]][self.idict[i]]))
       
        n_hour = datetime.datetime.now().strftime('%H')
        clu_list = [ clu for clu in all_n.keys() if clu[2:] == n_hour ]
       
        result_df = pd.DataFrame( [ [c, i, p] for c in clu_list for i, p in all_n[c] ], columns=['userID', 'itemID', 'pred_Score'])
        result_df['isChk'] = result_df['itemID'].apply(lambda x :  1 if x in allkill_list else 0 )
        result_df = result_df[result_df.isChk == 1][['userID', 'itemID', 'pred_Score']]
        result_df['rnk'] = result_df.groupby('userID')['pred_Score'].rank(method='first', ascending=False)
       
        return result_df
    
    
    def predict_newitem(self, model, item_list, item_features):
        
        i_dict = { i : idx for idx, i in enumerate(item_list)}
        
        for u in fm.ulist:
            scores = model.predict(fm.udict[u], np.arange(len(item_list)), item_features=item_features)
            all_scores = np.vstack((all_scores, scores))
    
    
        for u in fm.ulist:
            for i in item_list:
                all_n[u].append((i, all_scores[fm.udict[u]][i_dict[i]]))
                
        n_hour = datetime.datetime.now().strftime('%H')
        clu_list = [ clu for clu in all_n.keys() if clu[2:] == n_hour ]
        result_df = pd.DataFrame( [ [c, i, p] for c in clu_list for i, p in all_n[c] ], columns=['userID', 'itemID', 'pred_Score'])
        result_df['rnk'] = result_df.groupby('userID')['pred_Score'].rank(method='first', ascending=False)
        
        return result_df

    @staticmethod
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

    
    def set_AIRedis(self,_dfMerge, allkillManageItem):
        
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


            for str_z in ['10']:

                new_key = 'iac:personalization:deal:date:' + p_date + ':cluster:' + clusterno + str_z + ':itemnos'
#                 print(new_key, len(v))
                rc.delete(new_key)
                rc.rpush(new_key, *v)
                rc.expire(new_key,exp_sec)

    
def main():
    start = time.time()
    p_date = datetime.datetime.now().strftime('%Y%m%d')
    
    feature_list = [
        ['clk_1d_cnt', 'clk_2w_cnt', 'itemno'],
        ['ord_1d_cnt', 'ord_2w_cnt', 'itemno'],
        ['cr_repeat', 'cr_2w', 'itemno'],
        ['brand', 'itemno']
    ]
    ins_date = datetime.datetime.now().timestamp()
    weekno = datetime.datetime.now().weekday()

    features = None
    v = random.randint(0, len(feature_list) - 1)
    if v == 2:
        if weekno > 3:
            features = ['cr_repeat_wkend', 'cr_2w', 'itemno']
        else:
            features = ['cr_repeat_wkday', 'cr_2w', 'itemno']
    else:
        features = feature_list[v]

    print(str(features) + "|" + str(ins_date))
    #   fm = lightfm(features, 2, 30, 5, 1e-6)
    fm = lightfm(features)
    allkill_list, allkillManageItem = fm.getAllkillList()
    fm.getDruidOrd()
    fm.setFeatMapping(allkill_list)

    isNumeric = False if features[0] == 'brand' else True
    fm.getSqlOrd(fm.ilist, isNumeric)
    fm.make_item_features(isNumeric)
    fm.make_dataset(isNumeric)

    
    train, test = fm.split_train_test()
    model = fm.make_model(train)
  #  fm.calc_auc(model, train, test)
  #  allkill_list, allkillManageItem = fm.getAllkillList()
    result_df = fm.predict(model, allkill_list)
    fm.set_AIRedis(result_df, allkillManageItem)
  #  print(result_df.groupby(['userID'])['itemID'].nunique())
    print("time :", time.time() - start)
    
    
if __name__ == '__main__':
    main()