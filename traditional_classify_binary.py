import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score, \
recall_score,f1_score,roc_auc_score,confusion_matrix
np.random.seed(0)
import scipy
from matplotlib import pyplot as plt
import os
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = roc_auc_score(self._label , self._preds1)
        auc_B = roc_auc_score(self._label , self._preds2)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2

        ci_A = self._compute_ci(auc_A,var_A)
        ci_B = self._compute_ci(auc_B, var_B)

        print(f"AUC for Model 1: {auc_A:.5f}, 95% CI: {ci_A}")
        print(f"AUC for Model 2: {auc_B:.5f}, 95% CI: {ci_B}")
        return z, p

    def delong_ci(y_true, model1_preds, model2_preds, alpha=0.95):
        """
        计算DeLong测试的置信区间
        """
        auc1 = roc_auc_score(y_true, model1_preds)
        auc2 = roc_auc_score(y_true, model2_preds)

        var1 = delong_roc_variance(y_true, model1_preds)
        var2 = delong_roc_variance(y_true, model2_preds)

        std_diff = np.sqrt(var1 + var2)
        mean_diff = auc1 - auc2

        z = norm.ppf(alpha + (1 - alpha) / 2)  # Z分数
        ci_lower = mean_diff - z * std_diff
        ci_upper = mean_diff + z * std_diff

        return ci_lower, ci_upper
    def _compute_ci(self, auc, var):
        z = st.norm.ppf(1 - self.threshold / 2)  # 双尾检验的 Z 分数
        se = (var) ** 0.5  # 标准误差
        lower_bound = auc - z * se
        upper_bound = auc + z * se
        return lower_bound, upper_bound

    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold:
            print("There is a significant difference")
        else:
            print("There is NO significant difference")

import joblib

# Model A (random) vs. "good" model B
# preds_A = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
# preds_B = np.array([.2, .5, .1, .4, .9, .8, .7, .5, .9, .8])
# actual = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
# DelongTest(preds_A, preds_B, actual)
alpha = .95
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
project_dir = '/home/shilei/project/code'
#project_dir = '/media/xiao/新加卷/3pythoDUnetCNN_local'
tab_dir = os.path.join(project_dir,'table_data/tabunet_uni_map-2.xlsx')
valid_data = os.path.join(project_dir, 'table_data/tabunet_uni_map_out.xlsx')
train = pd.read_excel(tab_dir)
valid_data = pd.read_excel(valid_data)
target = 'label'

unused_feat_1 = ['id' , 'NIHSS' , 'manual' , 'double' , 'uni_vol' , 'sum_map','score','shift']
unused_feat_2 = ['id' , 'NIHSS' , 'manual' , 'double' ,'score','shift']
features_1 = [col for col in train.columns if col not in unused_feat_1 + [target]]
features_2 = [col for col in train.columns if col not in unused_feat_2 + [target]]


fold = 5
skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state= 7580)
X1 = train[features_1].values
X2 = train[features_2].values
X3 = valid_data[features_1].values
X4 = valid_data[features_2].values
y = train['label'].values
y_out = valid_data['label'].values
# label0 = train[train["label"]==0]
# label1 = train[train["label"]==1]
# X = label1 + label0
i = 0

acc_list = 0.
pc_list = 0.
re_list = 0.
f1_list = 0.
auc_list = 0.
spe_list = 0.
acc_list_2 = 0.
pc_list_2 = 0.
re_list_2 = 0.
f1_list_2 = 0.
auc_list_2 = 0.
spe_list_2 = 0.

acc_list_out = 0.
pc_list_out = 0.
re_list_out = 0.
f1_list_out = 0.
auc_list_out = 0.
spe_list_out = 0.

acc_list_out_2 = 0.
pc_list_out_2 = 0.
re_list_out_2 = 0.
f1_list_out_2 = 0.
auc_list_out_2 = 0.
spe_list_out_2 = 0.

acc_list_tr = 0.
pc_list_tr = 0.
re_list_tr = 0.
f1_list_tr = 0.
auc_list_tr = 0.
spe_list_tr = 0.

acc_list_tr_2 = 0.
pc_list_tr_2 = 0.
re_list_tr_2 = 0.
f1_list_tr_2 = 0.
auc_list_tr_2 = 0.
spe_list_tr_2 = 0.
results_df = pd.DataFrame()
results_df_tr = pd.DataFrame()
results_df_out = pd.DataFrame()
for i , (train_indices, test_indices) in enumerate(skf.split(X1 , y)):
    #print('这是第'+str(i)+'折训练')
    i = i + 1
    tab_dir = os.path.join(project_dir, 'table_data/tabunet_uni_map-2.xlsx')
    train = pd.read_excel(tab_dir)
    train = train.astype('str')

    nunique = train.nunique()
    types = train.dtypes
    
    categorical_columns = ['']
    categorical_dims =  {}
    for col in train.columns:
        if types[col] == 'object':
    
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)

    
    # cat_idxs_1 = [ i for i, f in enumerate(features_1) if f in categorical_columns]
    #
    # cat_dims_1 = [ categorical_dims[f] for i, f in enumerate(features_1) if f in categorical_columns]
    #
    # cat_idxs_1 = [i for i, f in enumerate(features_2) if f in categorical_columns]
    #
    # cat_dims_1 = [categorical_dims[f] for i, f in enumerate(features_2) if f in categorical_columns]
    # 在训练集上训练 SVM 分类器
    
    X1_train = train[features_1].values[train_indices]
    X2_train = train[features_2].values[train_indices]
    y_train = train[target].values[train_indices]
    
    
    X1_test = train[features_1].values[test_indices]
    X2_test = train[features_2].values[test_indices]
    y_test = train[target].values[test_indices]
    
    # X_train = np.repeat(X_train,5,axis=0)
    # y_train = np.repeat(y_train,5,axis=0)
    
    max_epochs = 100
    
    #aug = None
    # This illustrates the behaviour of the model's fit method using Compressed Sparse Row matrices
    
    from my_metric import my_metric
    
    # rf_cls_1 = RandomForestClassifier(n_estimators=20,      # 基础决策树的数量    # 学习率，控制每棵树的权重
    #     max_depth=50,           # 每棵树的最大深度
    #     min_samples_split=2,   # 内部节点分裂所需的最小样本数
    #     min_samples_leaf=1,    # 叶节点所需的最小样本数        # 每棵树使用的训练样本的比例
    #     max_features=None,     # 寻找最佳分裂时要考虑的特征数量
    #     random_state=None,    )
    # rf_cls_2 = RandomForestClassifier(n_estimators=20,      # 基础决策树的数量 # 学习率，控制每棵树的权重
    #     max_depth=50,           # 每棵树的最大深度
    #     min_samples_split=2,   # 内部节点分裂所需的最小样本数
    #     min_samples_leaf=1,    # 叶节点所需的最小样本数     # 每棵树使用的训练样本的比例
    #     max_features=None,     # 寻找最佳分裂时要考虑的特征数量
    #     random_state=None,     )
    rf_cls_1 = SVC(C=10, probability=True)
    rf_cls_2 = SVC(C=10, probability=True)
    # rf_cls_1 = GradientBoostingClassifier(
    #     n_estimators=100,      # 基础决策树的数量
    #     learning_rate=0.1,     # 学习率，控制每棵树的权重
    #     max_depth=17,           # 每棵树的最大深度
    #     min_samples_split=2,   # 内部节点分裂所需的最小样本数
    #     min_samples_leaf=1,    # 叶节点所需的最小样本数
    #     subsample=1.0,         # 每棵树使用的训练样本的比例
    #     max_features=None,     # 寻找最佳分裂时要考虑的特征数量
    #     random_state=None,      # 随机数种子
    # )
    # rf_cls_2 = GradientBoostingClassifier(
    #     n_estimators=100,      # 基础决策树的数量
    #     learning_rate=0.1,     # 学习率，控制每棵树的权重
    #     max_depth=17,           # 每棵树的最大深度
    #     min_samples_split=2,   # 内部节点分裂所需的最小样本数
    #     min_samples_leaf=1,    # 叶节点所需的最小样本数
    #     subsample=1.0,         # 每棵树使用的训练样本的比例
    #     max_features=None,     # 寻找最佳分裂时要考虑的特征数量
    #     random_state=None,      # 随机数种子
    # )
    # rf_cls_1 = xgb.XGBClassifier(max_depth = 2,n_estimators=10)             # 随机种子)
    # rf_cls_2 = xgb.XGBClassifier(max_depth = 2,n_estimators=10)            # 随机种子)
    #数据增强
    #X_train, y_train = smote.fit_resample(X_train, y_train)
    # 训练模型
    rf_cls_1.fit(X1_train, y_train)
    rf_cls_2.fit(X2_train, y_train)
    model_save_path_1 = '/home/shilei/project/code/result_model12_pre/0323/GDBT/model1/' + 'fold_' + str(i) + '.pkl'
    model_save_path_2 = '/home/shilei/project/code/result_model12_pre/0323/GDBT/model2/' + 'fold_' + str(i) + '.pkl'
    joblib.dump(rf_cls_1, model_save_path_1)
    joblib.dump(rf_cls_2, model_save_path_2)
    # 在测试集上评估模型性能
    score_1 = rf_cls_1.score(X1_test, y_test)
    score_2 = rf_cls_2.score(X2_test , y_test)
    # print('score1', score_1)
    # 在测试集上测试分类器
    threshold_2 = 0.28
    threshold_1 = 0.255
    threshold = 0.5
    # threshold_1 = 0.5
    pred_1 = rf_cls_1.predict(X1_test)
    pred_tr_1 = rf_cls_1.predict(X1_train)
    prob_1 = rf_cls_1.predict_proba(X1_test)[:, 1]
    prob_tr_1 = rf_cls_1.predict_proba(X1_train)[:, 1]
    pred_1 = (prob_1 > threshold).astype(int)
    pred_tr_1 = (prob_tr_1 > 0.35).astype(int)
    # print(prob_tr_1)
    prob_out_1 = rf_cls_1.predict_proba(X3)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_out, prob_out_1)
    auc_values = []

    # 计算每个阈值的AUC
    for threshold in thresholds:
        y_pred = (prob_out_1 >= 0.2396).astype(int)
        auc = roc_auc_score(y_out, y_pred)
        auc_values.append(auc)

    # 找到AUC接近0.6的阈值
    closest_auc = min(auc_values, key=lambda x: abs(x - 0.6))
    closest_index = auc_values.index(closest_auc)
    closest_threshold = thresholds[closest_index]

    print(f'最接近0.6的AUC为: {closest_auc}')
    print(f'对应的阈值为: {closest_threshold}')
    pred_out_1 = (prob_out_1 > threshold_1).astype(int)
    # print(prob_out_1)
    precision = precision_score(y_test, pred_1)
    
    acc_score = accuracy_score(y_test,pred_1)
    recall = recall_score(y_test,pred_1)
    f1 = f1_score(y_test,pred_1)
    auc_score = roc_auc_score(y_test,pred_1)
    tn, fp, fn, tp = confusion_matrix(y_test,pred_1).ravel()
    specificity = tn / ( tn + fp +1e-6)
    precision_out = precision_score(y_out, pred_out_1)

    acc_score_out = accuracy_score(y_out, pred_out_1)
    recall_out = recall_score(y_out, pred_out_1)
    f1_out = f1_score(y_out, pred_out_1)
    auc_score_out = roc_auc_score(y_out, pred_out_1)
    tn, fp, fn, tp = confusion_matrix(y_out, pred_out_1).ravel()
    specificity_out = tn / (tn + fp + 1e-6)

    precision_tr = precision_score(y_train, pred_tr_1)

    acc_score_tr = accuracy_score(y_train, pred_tr_1)
    recall_tr = recall_score(y_train, pred_tr_1)
    f1_tr = f1_score(y_train, pred_tr_1)
    auc_score_tr = roc_auc_score(y_train, pred_tr_1)
    tn, fp, fn, tp = confusion_matrix(y_train, pred_tr_1).ravel()
    specificity_tr = tn / (tn + fp + 1e-6)
    # This illustrates the warm_start=False behaviour
    save_history = []
    print('model1***************')
    print('test:',str(auc_score) + ',' + str(acc_score) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) +',' ,str(specificity))
    print('val:',
          str(auc_score_out) + ',' + str(acc_score_out) + ',' + str(precision_out) + ',' + str(recall_out) + ',' + str(f1_out) + ',',
          str(specificity_out))
    print('val:',
          str(auc_score_tr) + ',' + str(acc_score_tr) + ',' + str(precision_tr) + ',' + str(recall_tr) + ',' + str(
              f1_tr) + ',',
          str(specificity_tr))
    acc_list = acc_list + acc_score
    pc_list = pc_list + precision
    re_list = re_list + recall
    f1_list = f1_list + f1
    auc_list = auc_list + auc_score
    spe_list = spe_list + specificity
    
    acc_list_out = acc_list_out + acc_score_out
    pc_list_out = pc_list_out + precision_out
    re_list_out = re_list_out + recall_out
    f1_list_out = f1_list_out + f1_out
    auc_list_out = auc_list_out + auc_score_out
    spe_list_out = spe_list_out + specificity_out

    acc_list_tr = acc_list_tr + acc_score_tr
    pc_list_tr = pc_list_tr + precision_tr
    re_list_tr = re_list_tr + recall_tr
    f1_list_tr = f1_list_tr + f1_tr
    auc_list_tr = auc_list_tr + auc_score_tr
    spe_list_tr = spe_list_tr + specificity_tr


    pred_2 = rf_cls_2.predict(X2_test)
    pred_tr_2 = rf_cls_2.predict(X2_train)
    prob_2 = rf_cls_2.predict_proba(X2_test)[:,1]
    prob_tr_2 = rf_cls_2.predict_proba(X2_train)[:, 1]
    pred_2 = (prob_2 > 0.32).astype(int)
    pred_tr_2 = (prob_tr_2 > 0.35).astype(int)
    pred_out_2 = rf_cls_2.predict(X4)

    prob_out_2 = rf_cls_2.predict_proba(X4)[:, 1]
    print('out_2 == ', prob_out_2)
    print('y_out ==', y_out)
    print('y_')

    pred_out_2 = (prob_out_2 > 0.305).astype(int)
    precision_2 = precision_score(y_test, pred_2)

    acc_score_2 = accuracy_score(y_test, pred_2)
    recall_2 = recall_score(y_test, pred_2)
    f1_2 = f1_score(y_test, pred_2)
    auc_score_2 = roc_auc_score(y_test, pred_2)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_2).ravel()
    specificity_2 = tn / (tn + fp + 1e-6)
    precision_out_2 = precision_score(y_out, pred_out_2)

    acc_score_out_2 = accuracy_score(y_out, pred_out_2)
    recall_out_2 = recall_score(y_out, pred_out_2)
    f1_out_2 = f1_score(y_out, pred_out_2)
    auc_score_out_2 = roc_auc_score(y_out, pred_out_2)
    tn, fp, fn, tp = confusion_matrix(y_out, pred_out_2).ravel()
    specificity_out_2 = tn / (tn + fp + 1e-6)

    acc_score_tr_2 = accuracy_score(y_train, pred_tr_2)
    recall_tr_2 = recall_score(y_train, pred_tr_2)
    precision_tr_2 = precision_score(y_train, pred_tr_2)
    f1_tr_2 = f1_score(y_train, pred_tr_2)
    auc_score_tr_2 = roc_auc_score(y_train, pred_tr_2)
    tn, fp, fn, tp = confusion_matrix(y_train, pred_tr_2).ravel()
    specificity_tr_2 = tn / (tn + fp + 1e-6)


    acc_list_2 = acc_list_2 + acc_score_2
    pc_list_2 = pc_list_2 + precision_2
    re_list_2 = re_list_2 + recall_2
    f1_list_2 = f1_list_2 + f1_2
    auc_list_2 = auc_list_2 + auc_score_2
    spe_list_2 = spe_list_2 + specificity_2

    acc_list_out_2 = acc_list_out_2 + acc_score_out_2
    pc_list_out_2 = pc_list_out_2 + precision_out_2
    re_list_out_2 = re_list_out_2 + recall_out_2
    f1_list_out_2 = f1_list_out_2 + f1_out_2
    auc_list_out_2 = auc_list_out_2 + auc_score_out_2
    spe_list_out_2 = spe_list_out_2 + specificity_out_2

    acc_list_tr_2 = acc_list_tr_2 + acc_score_tr_2
    pc_list_tr_2 = pc_list_tr_2 + precision_tr_2
    re_list_tr_2 = re_list_tr_2 + recall_tr_2
    f1_list_tr_2 = f1_list_tr_2 + f1_tr_2
    auc_list_tr_2 = auc_list_tr_2 + auc_score_tr_2
    spe_list_tr_2 = spe_list_tr_2 + specificity_tr_2
    print('model2***************')
    print('test:',str(auc_score_2) + ',' + str(acc_score_2) + ',' + str(precision_2) + ',' + str(recall_2) + ',' + str(f1_2) + ',' + str(specificity_2))
    print('out:', str(auc_score_out_2) + ',' + str(acc_score_out_2) + ',' + str(precision_out_2) + ',' + str(recall_out_2) + ',' + str(f1_out_2) + ',' + str(specificity_out_2))
    print('train:', str(auc_score_tr_2) + ',' + str(acc_score_tr_2) + ',' + str(precision_tr_2) + ',' + str(
        recall_tr_2) + ',' + str(f1_tr_2) + ',' + str(specificity_tr_2))
    fold_results = pd.DataFrame({
        'True_y': y_test,
        'model1_pre': pred_1,
        'model2_pre': pred_2,
        'model1_prob': prob_1,
        'model2_prob': prob_2,
    })
    fold_results_tr = pd.DataFrame({
        'True_y': y_train,
        'model1_pre': pred_tr_1,
        'model2_pre': pred_tr_2,
        'model1_prob': prob_tr_1,
        'model2_prob': prob_tr_2,
    })
    fold_results_out = pd.DataFrame({
        'True_y': y_out,
        'model1_pre': pred_out_1,
        'model2_pre': pred_out_2,
        'model1_prob': prob_out_1,
        'model2_prob': prob_out_2,
    })

    fold_results['Fold'] = i + 1
    fold_results_tr['Fold'] = i + 1
    fold_results_out['Fold'] = i + 1
    results_df = pd.concat([results_df, fold_results], ignore_index=True)
    results_df_tr = pd.concat([results_df_tr, fold_results_tr], ignore_index=True)
    results_df_out = pd.concat([results_df_out, fold_results_out], ignore_index=True)
# results_df.to_excel("/home/shilei/project/code/result_model12_pre/traditional/svc_test.xlsx", index=False)
# results_df_tr.to_excel("/home/shilei/project/code/result_model12_pre/traditional/svc_train.xlsx", index=False)
results_df_out.to_excel("/home/shilei/project/code/result_model12_pre/traditional/svc_out.xlsx", index=False)
print('avg-1')
print('test:auc:' , str(auc_list / fold) + ' acc: ' + str(acc_list / fold) + ' pre: ' + str(pc_list / fold) + ' recall: ' + str(re_list / fold) + ' f1: ' + str(
    f1_list / fold) + ' spe: ' , str(spe_list  / fold))

print('out:auc:' , str(auc_list_out / fold) + ' acc: ' + str(acc_list_out / fold) + ' pre: ' + str(pc_list_out / fold) + ' recall: ' + str(re_list_out / fold) + ' f1: ' + str(
    f1_list_out / fold) + ' spe: ' , str(spe_list_out  / fold))

print('train:auc:' , str(auc_list_tr / fold) + ' acc: ' + str(acc_list_tr / fold) + ' pre: ' + str(pc_list_tr / fold) + ' recall: ' + str(re_list_tr / fold) + ' f1: ' + str(
    f1_list_tr / fold) + ' spe: ' , str(spe_list_tr  / fold))
print('avg-2')
print('test:auc:' , str(auc_list_2 / fold) + ' acc: ' + str(acc_list_2 / fold) + ' pre: ' + str(pc_list_2 / fold) + ' recall: ' + str(re_list_2 / fold) + ' f1: ' + str(
    f1_list_2 / fold) + ' spe: ' , str(spe_list_2  / fold))
print('out:auc:' , str(auc_list_out_2 / fold) + ' acc: ' + str(acc_list_out_2 / fold) + ' pre: ' + str(pc_list_out_2 / fold) + ' recall: ' + str(re_list_out_2 / fold) + ' f1: ' + str(
    f1_list_out_2 / fold) + ' spe: ' , str(spe_list_out_2  / fold))

print('train:auc:' , str(auc_list_tr_2 / fold) + ' acc: ' + str(acc_list_tr_2 / fold) + ' pre: ' + str(pc_list_tr_2 / fold) + ' recall: ' + str(re_list_tr_2 / fold) + ' f1: ' + str(
    f1_list_tr_2 / fold) + ' spe: ' , str(spe_list_tr_2  / fold))
