import concurrent
import csv
import datetime
from scipy import stats
import hyperopt
import joblib
from hyperopt import fmin, tpe
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score, precision_score, \
    recall_score, average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import copy
import warnings
warnings.filterwarnings('ignore')

"""
this classifier is built based on the stacking of several popular boosting methods (also support not boosting methods):
        included base learner "child_learner": [adaboost, gbm, xgb, lgb, catboost,svc,lr,rf,et,mlp]
the outputs (predictions of probabilities) of these base learner are input features of one parent learner:
        included parent learner "parent_learner": one of [adaboost, gbm, xgb, lgb, catboost,svc,lr,rf,et,mlp]
we support different stacking method by setting the "method" parameter in fit(), predict() or predict_prob():
   support using class as predictions by setting it as "class",
   support using probabilities as predictions by setting it as "proba",
   support using the max probability as prediction by setting it as "maxProba",
   support averaging method by setting it as "average",
   support simple voting method by setting it as "voting",
   support weighted voting method by setting it as "weighted_average", weights are assigned according to the prediction accuracy
   support using the best base learner by setting it as "best"
we also support stacking with the raw features by setting  "with_raw_features=True",
we use both wrapping and embedding method to select important features,
    the  function is built upon RFECV() and SelectFromModel() in scikit-learn,
    RFECV() is a wrapping method, while SelectFromModel() is a embedding method :
        tune_feature = None|"default"|[{"estimator": estimator,"tf_manner": None or "wrapping" or "embedding"},...]
        None: do not perform feature selection
        "default": perform feature selection for all the base learners specified in child_learner,
        [{"estimator": estimator,"tf_manner": None or "wrapping" or "embedding"},...]:
            perform feature selection for the specified estimator with specified feature tuning manner
            "wrapping" means using RFECV(), while "embedding" means using SelectFromModel()
the parameter tuning function is built upon GridSearchV, RandomlizedSearchCV, BayseSearchCV, and hyperopt module:
        child_optimizer: None|"default"|[{"estimator": AdaBoostClassifier(), "params":params, "tp_manner":"gs"},...]
        parent_optimizer = None|"default"|{"params":params, "tp_manner":"gs","tf_manner":"wrapping"}
        None: do not perform parameter tuning
        "default": perform parameter tuning for every base learner specified in child_learner,
        [{"estimator": AdaBoostClassifier(), "params":params, "tp_manner":"gs"},...]:
                    a list of dict (a dict for parent_optimizer) contains
                    "estimator": estimator names, include adaboost, gbm, xgb, lgb, catboost,rf,svc,lr
                    "params": parameter search space, a dict, with parameter names as keys and search spaces as values
                    "tp_manner": search manners, include gs, bayes, hpopt, random
                    "tf_manner": only available for parent_optimizer, include values [None, "default","wrapping","embedding"]
        if you do not want to perform parameter tuning, just leave the child_optimizer or parent_optimizer as None,the default setting
        if you want to use the default tuning, set "default" to them,
        if you want costomized tuning, you need to set them as described above
        remember that the tuning process can be long, and the "parent_optimizer" is a dict instead of a list of dict
we also support single model pattern with "single" parameter: 
    it's default is None which means stacking pattern, otherwise, set it as a dict with three items
    single["estimator"]: one of [rf,adaboost, gbm, xgb, lgb, catboost,svc,lr]
    single["tf"]: one of [None,"default","wrapping","embedding"], None means no feature selection,
    single["tp"]: one of [None,"default",{"params":params space,"tp_manner": one of "bayes,hpopt,random,gs"}], 
                    None means no parameter tuning, 
                    "default" means default tuning, 
                    customized tuning is supported by setting a dict with parameter space and tuning manner
lightgbm and catboost support categorical features with categorical column indices,
when dealing with imbalanced problem, it's better to set the "class_weight" parameter as "balanced" if the learner has one
estimators with "class_weight" parameter:
                      ExtraTreesClassifier(), SVC(),LogisticRegression(),RandomForestClassifier(),LGBMClassifier()
For multiclass problem, you need to set the "loss_function" parameter in CatBoostClassifier() as "Multiclass" if you include it
We also support SVC(), LogisticRegression(),MLPClassifier() and other algorithms might be enclosed in the future
The reason we use hyperopt for catboost is that we couldn't fix the bug when using it with bayes optimizer
    and some research shows that hyperopt is faster than scikit-optimizer, but the latter contains more optimizing algorithms
The scoring function can be tricky, "roc_auc" is not supported for multiclass problem and sometimes "neg_log_loss" (this is why it's tricky)
    especially when your data is small and some labels are not predicted
Non-linear SVC() and MLP can't not be used with feature selection, cause their do not expose coef_ or feature_importantance
"""

# we use [adaboost, gbm, xgb, lgb, catboost] as default child_learner, and "rf" as default parent_learner
DEFAULT_CHILD_LEARNER =[
        AdaBoostClassifier(base_estimator=RandomForestClassifier(verbose=2,n_estimators=20)),
        GradientBoostingClassifier(verbose=2),
        RandomForestClassifier(verbose=2,n_estimators=100,class_weight="balanced"),
        XGBClassifier(),LGBMClassifier(class_weight="balanced"),
        CatBoostClassifier(verbose=2,loss_function='MultiClass'),
        ExtraTreesClassifier(verbose=2,n_estimators=100,class_weight="balanced"),
        SVC(class_weight="balanced",verbose=2,probability=True,kernel="linear"),
        LogisticRegression(class_weight="balanced",verbose=2)
               ]
DEFAULT_PARENT_LEARNER = RandomForestClassifier(class_weight="balanced",verbose=2,n_estimators=100)

class StackingBoostClassifier():
    def __init__(self,
                 child_learner=DEFAULT_CHILD_LEARNER,
                 parent_learner=DEFAULT_PARENT_LEARNER,
                 single=None,
                 tune_feature = None,child_optimizer=None,
                 parent_optimizer=None,cat_features=None,with_raw_features=False):
        self.child_learner = child_learner
        self.parent_learner = parent_learner
        self.selector_parent = None
        self.tune_feature = tune_feature
        self.child_optimizer = child_optimizer
        self.parent_optimizer = parent_optimizer
        # we can specify the categorical feature indices for lightgbm and catboost
        self.cat_features = cat_features
        # this is used to store the fitted child_learner, i.e. {"xgb":XGBoost().fit(X,y),...}
        self.fited_child_learner = {}
        self.weighted_child_learner = {}
        # this is used to store the feature selectors for the child_learners, i.e. {"xgb":XGBoost(),...}
        self.selector = {}
        self.single= single
        self.selector_single = None
        self.estimator_single = None
        self.with_raw_features=with_raw_features
    def set_params(self,param_dict):
        for k, v in param_dict.items():
            if k=="child_learner":
                self.child_learner = v
            if k=="parent_learner":
                self.parent_learner = v
            if k=="single":
                self.single = v
            if k == "tune_feature":
                self.tune_feature = v
            if k== "child_optimizer":
                self.child_optimizer = v
            if k == "parent_optimizer":
                self.parent_optimizer = v
            if k == "cat_features":
                self.cat_features = v
            if k == "with_raw_features":
                self.with_raw_features = v
    # this function is used to tune paramters,
    # we need to specify the estimator, tuning manner, parameter space, X, y, and score metric
    # we save the best tuned model and return a dict with the name as key and the best model as value
    def tune_parameter(self,estimator,tp_manner,params,X,y,scoring="neg_log_loss"):
        estimator_name = (self.get_default_params_and_name(estimator))[0]
        print("tune parameters for "+estimator_name)
        if tp_manner == "bayes":
            if estimator_name in ["rf","et"]:
                base_estimator = "RF"
            elif estimator_name in ["adaboost","xgb","lgb","gbm","catboost"]:
                base_estimator = "GBRT"
            else:
                base_estimator = "GP"
            tp = BayesSearchCV(
                estimator=estimator, search_spaces=params,optimizer_kwargs={"base_estimator":base_estimator},
                scoring=scoring, n_iter=60,
                verbose=2, n_jobs=-1, cv=3, refit=True, random_state=1234
            )
        elif tp_manner == "gs":
            tp = GridSearchCV(
                estimator=estimator, param_grid=params, scoring=scoring,
                n_jobs=-1, cv=3, refit=True, verbose=2
            )
        elif tp_manner == "random":
            tp = RandomizedSearchCV(
                estimator=estimator, param_distributions=params,
                scoring=scoring, n_jobs=-1, n_iter=60,
                cv=3, refit=True, verbose=2, random_state=1234)
        elif tp_manner == "hpopt":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)
            space = params
            def objective(space):
                clf = estimator
                clf.set_params(**space)
                clf.fit(X=X_train,y=y_train)
                loss = self.get_loss(clf,X_test,y_test,scoring)
                return loss
            best_param = fmin(fn=objective,
                              space=space,
                              algo=tpe.suggest,
                              max_evals=60)

            str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
            tp = estimator.set_params(**best_param)
            tp.fit(X,y)
            y_hat = tp.predict(X)
            metrics_dict = self.get_metrics(y_hat,y)
            print(estimator_name,best_param)
            print(metrics_dict)
            model_name = estimator_name+str_time+".pkl"
            print("save metrics to tp_log.csv:" , estimator_name)
            with open("tp_log.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([estimator_name,model_name," best params : ", str_time])
                for key, value in metrics_dict.items():
                    writer.writerow([key, value])
                for key, value in best_param.items():
                    writer.writerow([key, value])
            # tp.save_model(model_name)
            joblib.dump(tp, model_name)
            return {estimator_name:tp}
        else:
            #todo
            return
        if estimator_name == "catboost":
            tp.fit(X=X,y=y,cat_features=self.cat_features)
        elif estimator_name == "lgb" and self.cat_features:
            tp.fit(X=X, y=y, categorical_feature=self.cat_features)
        else:
            tp.fit(X, y)
        best_param = tp.best_params_
        best_score = tp.best_score_
        y_hat = tp.predict(X)
        metrics_dict = self.get_metrics(y_hat, y)
        print(estimator_name,best_param)
        print("best score:",best_score)
        print(metrics_dict)
        str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        model_name = estimator_name + str_time + ".pkl"
        print("save metrics to tp_log.csv:",estimator_name)
        str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        with open("tp_log.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([estimator_name,model_name," best params : ",str_time])
            for key, value in metrics_dict.items():
                writer.writerow([key, value])
            for key, value in best_param.items():
                writer.writerow([key, value])
        joblib.dump(tp, model_name)
        return {estimator_name:tp}

    # this function is used to get the test loss score of an fitted estimator, the lower, the better
    def get_loss(self,estimator,X,y,scoring):
        loss = float("inf")
        pred = estimator.predict(X)
        pred = np.reshape(pred, -1)
        if scoring == "neg_log_loss":
            pred = estimator.predict_proba(X)
            loss = log_loss(y, pred)
        elif scoring == "roc_auc":
            loss = -roc_auc_score(y, pred)
        elif scoring == "f1_weighted":
            loss = -f1_score(y, pred, average="weighted")
        elif scoring == "f1":
            loss = -f1_score(y, pred)
        elif scoring == "accuracy":
            loss = -accuracy_score(y, pred)
        elif scoring == "precision_weighted":
            loss = -precision_score(y, pred, average="weighted")
        elif scoring == "average_precision":
            loss = -average_precision_score(y, pred)
        elif scoring == "precision":
            loss = -precision_score(y, pred)
        elif scoring == "recall":
            loss = -recall_score(y, pred)
        elif scoring == "recall_weighted":
            loss = -recall_score(y, pred, average="weighted")
        return loss
    def get_metrics(self,y_hat,y):
        metric_dict = {}
        acc = accuracy_score(y, y_hat)
        metric_dict.update({"accuracy":acc})
        recall = recall_score(y, y_hat, average='weighted')
        metric_dict.update({"recall_weighted": recall})
        f1 = f1_score(y, y_hat, average='weighted')
        metric_dict.update({"f1_weighted": f1})
        precision = precision_score(y, y_hat, average='weighted')
        metric_dict.update({"precision_weighted": precision})
        return metric_dict

    # this function is used to tune the threshold to select important features for an estimator
    # retuen a dict with the name of the estimator as key and the selector as value
    def select_feature_wrapping(self,estimator,X,y,scoring):
        estimator_name = (self.get_default_params_and_name(estimator))[0]
        print("using recursive feature elimination to tune features: "+estimator_name)
        selector = RFECV(estimator, step=1, cv=3,scoring=scoring,verbose=2)
        selector = selector.fit(X, y)
        sn = selector.n_features_
        sc = selector.score(X,y)
        sr = selector.ranking_
        print("features number and score:",sn,sc)
        print("selected features ranking:",sr)
        with open("tf_log.csv",'a',newline='') as f:
            writer = csv.writer(f)
            str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
            writer.writerow(["feature selection with rfecv ",estimator_name, str_time])
            writer.writerow(["feature selection score: ",sc,"selected feature number:", sn, "feature ranking:"])
            writer.writerow(sr)
        return {estimator_name:selector}
    def select_feature_embedding(self,estimator,X,y,scoring):
        estimator_name = (self.get_default_params_and_name(estimator))[0]
        print("using feature_importance or coef_ attributes of the estimator to tune features: "+estimator_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)
        clf = copy.deepcopy(estimator)
        clf.fit(X_train, y_train)
        lloss_ls = []
        mean_ls = [i/100.0 for i in range(0,120,5)]
        # mean_ls = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        for i in mean_ls:
            thres = str(i)+"*mean"
            sel = SelectFromModel(clf, prefit=True,threshold=thres)
            X_new_train = sel.transform(X_train)
            X_new_test = sel.transform(X_test)
            # X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y, test_size=0.3, random_state=42,shuffle=False)
            clf_val = copy.deepcopy(estimator)
            clf_val.fit(X_new_train,y_train)
            loss = self.get_loss(clf_val,X_new_test,y_test,scoring)
            # print(thres,loss)
            lloss_ls.append(loss)
        loss_mean_tuple = sorted(zip(lloss_ls,mean_ls),key= lambda x:x[0])
        # print(loss_mean_tuple)
        min_loss = loss_mean_tuple[0][0]
        min_loss_mean = loss_mean_tuple[0][1]
        threshold = str(min_loss_mean) + "*mean"
        print("threshold and min loss of embedding feature selection ",threshold,min_loss)
        print("loss threshold tuples:", loss_mean_tuple)
        selector = SelectFromModel(clf, prefit=True, threshold=threshold)
        with open("tf_log.csv",'a',newline='') as f:
            writer = csv.writer(f)
            str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
            writer.writerow(["embedding feature selection",estimator_name,str_time ])
            writer.writerow(["best threshold mean ratio and min loss: ", threshold,min_loss, "loss threshold tuple:"])
            writer.writerow(loss_mean_tuple)
        return {estimator_name:selector}

    def select_best_feature(self,estimator,X,y,scoring,tf_manner=None):
        estimator_name = (self.get_default_params_and_name(estimator))[0]
        if tf_manner =="default":
            if estimator_name in ["catboost","svc","gbm"]:
                return self.select_feature_embedding(estimator, X, y, scoring)
            else:
                return self.select_feature_wrapping(estimator,X,y,scoring)
        elif tf_manner == "wrapping":
            return self.select_feature_wrapping(estimator, X, y, scoring)
        elif tf_manner == "embedding":
            return self.select_feature_embedding(estimator, X, y, scoring)
        else:#todo
            return self.select_feature_wrapping(estimator,X,y,scoring)


    # this function is used to get the parameter space and the string name tuple for an estimator
    def get_default_params_and_name(self,estimator):
        # print("get default estimator name and parameters")
        if isinstance(estimator,AdaBoostClassifier):
            base_estimator = [
                RandomForestClassifier(verbose=2, n_estimators=20),
                ExtraTreesClassifier(verbose=2,n_estimators=20),
                # GradientBoostingClassifier(verbose=2,n_estimators=20),
                # XGBClassifier(n_estimators=20),
                # LGBMClassifier(n_estimators=20)
            ]
            params = {
                "n_estimators": (30, 200),
                "learning_rate": (1e-6, 1.0, 'log-uniform'),
                "base_estimator": Categorical(base_estimator),
            }
            estimator_name = "adaboost"
        elif isinstance(estimator,GradientBoostingClassifier):
            params = {
                "n_estimators": (300, 1200),
                "learning_rate": (1e-6, 1.0, 'log-uniform'),
                "max_depth": (3, 30),
                "min_samples_leaf": (1, 128),
                "min_samples_split": (2, 256),
                "subsample": (0.6, 1.0, 'uniform'),
                "max_features": (0.6, 1.0, 'uniform'),
                "min_weight_fraction_leaf": (0.0, 0.5, 'uniform'),
                "min_impurity_decrease": (1e-6, 1e-1, 'log-uniform'),
            }
            estimator_name = "gbm"
        elif isinstance(estimator,XGBClassifier):
            params = {
                "n_estimators": (300, 1200),
                "max_depth": (3, 30),
                "min_child_weight": (1e-3, 1e+3, 'log-uniform'),
                "learning_rate": (1e-6, 1.0, 'log-uniform'),
                "colsample_bytree": (0.6, 1.0, 'uniform'),
                "subsample": (0.6, 1.0, 'uniform'),
                "gamma": (1e-6, 1.0, 'log-uniform'),
                'reg_alpha': (1e-3, 1e3, 'log-uniform'),
                'reg_lambda': (1e-3, 1e3, 'log-uniform'),
                "scale_pos_weight": (0.01, 1.0, 'uniform'),
            }
            estimator_name = "xgb"
        elif isinstance(estimator,LGBMClassifier):
            params = {
                "n_estimators": (300, 1200),
                "max_depth": (3, 30),
                "max_bin": (64, 256),
                "num_leaves": (30, 256),
                "min_child_weight": (1e-3, 1e3, 'log-uniform'),
                "min_child_samples": (8, 256),
                "min_split_gain": (1e-6, 1.0, 'log-uniform'),
                "learning_rate": (1e-6, 1.0, 'log-uniform'),
                "colsample_bytree": (0.6, 1.0, 'uniform'),
                "subsample": (0.6, 1.0, 'uniform'),
                'reg_alpha': (1e-3, 1e3, 'log-uniform'),
                'reg_lambda': (1e-3, 1e3, 'log-uniform'),
                "scale_pos_weight": (0.01, 1.0, 'uniform'),
            }
            estimator_name = "lgb"
        elif isinstance(estimator,CatBoostClassifier):
            params = {
                # 'iterations': hyperopt.hp.quniform("iterations", 300, 1200, 10),
                'depth': hyperopt.hp.quniform("depth", 3, 12, 1),
                # 'border_count': hyperopt.hp.quniform("border_count", 16, 224, 4),
                'learning_rate': hyperopt.hp.loguniform('learning_rate', 1e-6, 1e-1),
                'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 3, 1),
                'bagging_temperature': hyperopt.hp.uniform('bagging_temperature', 0.6, 1.0),
                'rsm': hyperopt.hp.uniform('rsm', 0.8, 1.0)
            }
            estimator_name = "catboost"
        elif isinstance(estimator,RandomForestClassifier):
            params = {
                "n_estimators": (100, 1000),
                "criterion": Categorical(["gini", "entropy"]),
                "max_features": (0.8, 1.0, 'uniform'),
                "max_depth": (3, 30),
                "min_samples_split": (2, 256),
                "min_samples_leaf": (1, 128),
                "min_weight_fraction_leaf": (0.0, 0.5, 'uniform'),
                "max_leaf_nodes": (30, 256),
                "min_impurity_decrease": (1e-6, 1e-1, 'log-uniform'),
            }
            estimator_name = "rf"
        elif isinstance(estimator,ExtraTreesClassifier):
            params = {
                "n_estimators": (100, 1000),
                "criterion": Categorical(["gini","entropy"]),
                "max_features": (0.8, 1.0, 'uniform'),
                "max_depth": (3, 30),
                "min_samples_split": (2, 256),
                "min_samples_leaf": (1, 128),
                "min_weight_fraction_leaf": (0.0, 0.5, 'uniform'),
                "max_leaf_nodes": (30, 256),
                "min_impurity_decrease": (1e-6, 1e-1, 'log-uniform'),
            }
            estimator_name = "et"
        elif isinstance(estimator,SVC):
            params = {
                "C":  Real(1e-6, 1e+6, prior='log-uniform'),
                "gamma": Real(1e-6, 1e+1, prior='log-uniform'),
                "degree": Integer(1,3),
                "kernel": Categorical(['linear', 'poly', 'rbf']),
            }
            estimator_name = "svc"
        elif isinstance(estimator,LogisticRegression):
            params = {
                "C":  Real(1e-6, 1e+6, prior='log-uniform'),
                # "penalty": Categorical(['l1', 'l2']),
                "solver": Categorical(["newton-cg", "lbfgs", "liblinear", "saga"]),
            }
            estimator_name = "lr"
        elif isinstance(estimator,MLPClassifier):
            hls = []
            for i in [16, 32, 64]:
                hls.append((i * 2,i * 3))
                hls.append((i*2, i * 3, i*2))
                hls.append((i,i * 2, i * 4, i * 3))
            params = {
                "hidden_layer_sizes": Categorical(hls),
                "activation": Categorical(["logistic", "tanh", "relu"]),
                "solver": Categorical(["lbfgs", "sgd", "adam"]),
                "learning_rate": Categorical(["invscaling", "adaptive"]),
                "alpha": Categorical([0.00001, 0.0001, 0.001, 0.01]),
            }
            estimator_name = "mlp"
        else:
            print("wrong base estimator used")
            return
        return (estimator_name,params)

    # this function is used to select the important features from X for an estimator tuned by the select_best_feature() function
    def get_X_sel(self,X,estimator_name):
        if self.selector== None:
            X_sel = X
        elif self.selector[estimator_name]==None:
            X_sel = X
        else:
            X_sel = self.selector[estimator_name].transform(X)
        return X_sel

    def get_X_sel_single(self, X,selector):
        if selector == None:
            X_sel = X
        else:
            sel, = selector.values()
            # print(sel)
            # print(X.shape)
            X_sel = sel.transform(X)
            # print(X_sel.shape)
        return X_sel
    # this is an auxiliary function to get the outputs of the base learner
    # and transforms them as the input features for the parent learner
    def get_X_parent(self, X,method="maxProba"):
        # None for simple stacking, feature_weighted for feature weighted stacking, voting for simple voting
        # voting_weighted for weighted voting, best for using the best base learner
        df_X_class = pd.DataFrame()
        X_proba = []
        X_maxProba = []
        X_average = 0
        X_weighted_average = 0
        n_baselearner = 0
        for estimator_name, estimator in self.fited_child_learner.items():
            n_baselearner+=1
            X_sel = self.get_X_sel(X, estimator_name)
            print("get the prediction of ",estimator_name)
            if method in ["class","voting"]:
                y_class = estimator.predict(X_sel)
                y_class = np.reshape(y_class, -1)
                df_X_class[estimator_name] = pd.Series(y_class)
            elif method=="proba":
                y_proba = estimator.predict_proba(X_sel)
                X_proba.append(y_proba)
            elif method == "maxProba":
                y_proba = estimator.predict_proba(X_sel)
                y_maxProba = y_proba.max(axis=1)
                X_maxProba.append(y_maxProba)
            elif method == "average":
                y_proba = estimator.predict_proba(X_sel)
                # print(y_proba.shape)
                X_average+=y_proba
            elif method == "weighted_average":
                y_proba = estimator.predict_proba(X_sel)
                ratio = self.weighted_child_learner.get(estimator_name)
                # print(y_proba)
                # print(ratio)
                y_temp = y_proba*ratio
                X_weighted_average+=y_temp
        if method=="class":
            if self.with_raw_features:
                X_hat = np.concatenate((X, df_X_class.values), axis=1)
            else:
                X_hat = df_X_class.values
            return X_hat
        elif method=="proba":
            if self.with_raw_features:
                X_hat = np.concatenate((X, *X_proba), axis=1)
            else:
                X_hat = np.concatenate(X_proba,axis=1)
            return X_hat
        elif method=="maxProba":
            X_maxProba = np.array(X_maxProba)
            X_maxProba = X_maxProba.T
            if self.with_raw_features:
                X_hat = np.concatenate((X, X_maxProba ), axis=1)
            else:
                X_hat = X_maxProba
            return X_hat
        elif method =="average":
            return X_average/(1.0*n_baselearner)
        elif method == "weighted_average":
            return X_weighted_average
        elif method == "voting":
            y_mode = stats.mode(df_X_class.values, axis=1)
            y_hat = np.reshape(y_mode[0], -1)
            return y_hat
        elif method == "best":
            best_estimator_name = max(self.weighted_child_learner, key=self.weighted_child_learner.get)
            best_estimator = self.fited_child_learner.get(best_estimator_name)
            y_hat = best_estimator.predict(X)
            y_hat = np.reshape(y_hat, -1)
            return y_hat
        else:
            print("use the right method keyword")
            return
    # used to tune feature concurrently, base_learner is list of dict,
    # each dict contains the estimator and feature tuning manner
    def tf_concurrent(self, X, y, scoring,base_learner):
        with ProcessPoolExecutor(max_workers=7) as Executor:
            # process_pool = Executor(max_workers=5)
            tfps = []
            for base in base_learner:
                estimator = base.get("estimator")
                tf_manner = base.get("tf_manner")
                tfp = Executor.submit(self.select_best_feature, estimator, X, y, scoring,tf_manner)
                tfps.append(tfp)
            for tfp in concurrent.futures.as_completed(tfps):
                try:
                    res = tfp.result()
                    self.selector.update(res)
                    print(res)
                except Exception as e:
                    print('feature tuning process error. ' + str(e))
                else:
                    print('feature tuning process ok.')
        print('feature tuning finished!')
    # tune the feature
    def tf(self,X,y,scoring):
        if self.tune_feature ==None:
            self.selector = None
        elif self.tune_feature=="default":
            base_learner = []
            for estimator in self.child_learner:
                estimator_name = (self.get_default_params_and_name(estimator))[0]
                self.selector.update({estimator_name:None})
                base = {"estimator":estimator,"tf_manner":"default"}
                base_learner.append(base)
            self.tf_concurrent(X,y,scoring,base_learner)
        else:
            for estimator in self.child_learner:
                estimator_name = (self.get_default_params_and_name(estimator))[0]
                self.selector.update({estimator_name:None})
            self.tf_concurrent(X, y, scoring, self.tune_feature)
        print(self.selector)

    # fitting for a given estimator,return a dict with estimator name as key, and fitted estimator as value
    def plain_fit(self,estimator_name,estimator,X, y):
        if estimator_name == "catboost":
            fitted_estimator = estimator.fit(X, y, cat_features=self.cat_features)
            res = {estimator_name: fitted_estimator }
        elif estimator_name == "lgb" and self.cat_features:
            fitted_estimator = estimator.fit(X, y, categorical_feature=self.cat_features)
            res = {estimator_name: fitted_estimator }
        else:
            fitted_estimator = estimator.fit(X, y)
            res = {estimator_name: fitted_estimator}
        y_hat = fitted_estimator.predict(X)
        metrics_dict = self.get_metrics(y_hat,y)
        str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        with open("tp_log.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["base estimator, without tune parameter: ", estimator_name, str_time])
            for key, value in metrics_dict.items():
                writer.writerow([key, value])
        return res
    # cocurrent fitting for learners in given base_learner, used for non-tuning fitting
    def plain_fit_concurrent(self,X,y,base_learner):
        tpps = []
        with ProcessPoolExecutor(max_workers=7) as Executor:
            for estimator in base_learner:
                estimator_name = (self.get_default_params_and_name(estimator))[0]
                X_sel = self.get_X_sel(X, estimator_name)
                tpp = Executor.submit(self.plain_fit, estimator_name, estimator, X_sel, y)
                tpps.append(tpp)
            for tpp in concurrent.futures.as_completed(tpps):
                try:
                    res = tpp.result()
                    self.fited_child_learner.update(res)
                    print(res)
                except Exception as e:
                    print('model fit process error. ' + str(e))
                else:
                    print('model fit process ok.')
            print('model fitting finished!')
    def tp_fit_concurrent(self,X, y, scoring,base_learner,default=True):
        with ProcessPoolExecutor(max_workers=7) as Executor:
            # process_pool = Executor(max_workers=5)
            tpps = []
            for child in base_learner:
                if default:
                    name_param_tp = self.get_default_params_and_name(child)
                    estimator = child
                    estimator_name, params = name_param_tp[0], name_param_tp[1]
                    if estimator_name != "catboost":
                        tp_manner = "bayes"
                        X_sel = self.get_X_sel(X, estimator_name)
                        tpp = Executor.submit(self.tune_parameter, estimator, tp_manner, params, X_sel, y, scoring)
                        tpps.append(tpp)
                else:
                    estimator = child.get("estimator")
                    estimator_name = (self.get_default_params_and_name(estimator))[0]
                    tp_manner = child.get("tp_manner")
                    params = child.get("params")
                    X_sel = self.get_X_sel(X, estimator_name)
                    tpp = Executor.submit(self.tune_parameter, estimator, tp_manner, params, X_sel, y, scoring)
                    tpps.append(tpp)
            for tpp in concurrent.futures.as_completed(tpps):
                try:
                    res = tpp.result()
                    self.fited_child_learner.update(res)
                    print(res)
                except Exception as e:
                    print('parameter tuning process error. ' + str(e))
                else:
                    print('parameter tuning process ok.')
            print('parameter tuning finished!')
    # tune the parameters
    def tp(self,X,y,scoring):
        if self.child_optimizer == None:
            self.plain_fit_concurrent(X,y,self.child_learner)
        elif self.child_optimizer == "default":
            self.plain_fit_concurrent(X, y, self.child_learner)
            self.tp_fit_concurrent(X, y,scoring, self.child_learner,True)
        else:
            self.plain_fit_concurrent(X, y, self.child_learner)
            self.tp_fit_concurrent(X, y, scoring, self.child_optimizer, False)
        print(self.fited_child_learner)
    # begin to train the parent learner in stacking mode
    def stacking_mode(self,X,y,scoring="neg_log_loss",method="maxProba"):
        print("stacking mode, fit the parent estimator.")
        # if len(self.fited_child_learner) == 1:
        #     return
        X_parent = self.get_X_parent(X,method=method)
        if self.parent_optimizer==None:
            self.selector_parent =None
            self.parent_learner.fit(X_parent, y)
        elif self.parent_optimizer == "default":
            self.selector_parent  = self.select_best_feature(self.parent_learner, X_parent, y, scoring,"default")
            # print(X_parent.shape)
            # print(self.selector_parent)
            X_sel = self.get_X_sel_single(X_parent, self.selector_parent)
            # print(X_sel.shape)
            name_param_tp = self.get_default_params_and_name(self.parent_learner)
            estimator_name, params = name_param_tp[0], name_param_tp[1]
            if estimator_name == "catboost":
                name_estimator_dict = self.tune_parameter(self.parent_learner, "hpopt", params, X_sel, y,
                                                          scoring)
            else:
                name_estimator_dict = self.tune_parameter(self.parent_learner, "bayes", params, X_sel, y,
                                                          scoring)
            self.parent_learner = name_estimator_dict[estimator_name]
        else:
            params = self.parent_optimizer.get("params")
            tf_manner = self.parent_optimizer.get("tf_manner")
            tp_manner = self.parent_optimizer.get("tp_manner")
            if tf_manner== "wrapping":
                self.selector_parent = self.select_feature_wrapping(self.parent_learner, X_parent , y, scoring)
            elif tf_manner == "embedding":
                self.selector_parent = self.select_feature_embedding(self.parent_learner, X_parent , y, scoring)
            elif tf_manner == "default":
                self.selector_parent = self.select_best_feature(self.parent_learner, X_parent , y, scoring,"default")
            else:
                self.selector_parent = None
            # print(X_parent.shape)
            # print(self.selector_parent)
            X_sel = self.get_X_sel_single(X_parent, self.selector_parent)
            # print(X_sel.shape)
            estimator_name = (self.get_default_params_and_name(self.parent_learner))[0]
            if tp_manner == None:
                self.parent_learner.fit(X_sel, y)
            elif tp_manner  == "default":
                params = (self.get_default_params_and_name(self.parent_learner))[1]
                if estimator_name == "catboost":
                    name_estimator_dict = self.tune_parameter(self.parent_learner, "hpopt", params, X_sel, y,
                                                              scoring)
                else:
                    name_estimator_dict = self.tune_parameter(self.parent_learner, "bayes", params, X_sel, y,
                                                              scoring)
                self.parent_learner = name_estimator_dict[estimator_name]
            else:
                name_estimator_dict = self.tune_parameter(self.parent_learner, tp_manner, params, X_sel, y, scoring)
                self.parent_learner = name_estimator_dict[estimator_name]
    # fit in single estimator mode
    def single_mode(self,X,y,scoring):
       print("single mode, fit the single estimator.")
       estimator = self.single["estimator"]
       self.estimator_single = estimator
       estimator_name = (self.get_default_params_and_name(estimator))[0]
       tf = self.single["tf"]
       tp = self.single["tp"]
       if tf == "default":
           self.selector_single = self.select_best_feature(estimator, X, y, scoring)
       elif tf == "wrapping":
           self.selector_single = self.select_feature_wrapping(estimator, X, y, scoring)
       elif tf == "embedding":
           self.selector_single = self.select_feature_embedding(estimator, X, y, scoring)
       else:
           self.selector_single = None
       X_single = self.get_X_sel_single(X,self.selector_single)
       if tp == None:
           if estimator_name == "catboost":
               self.estimator_single.fit(X_single, y, cat_features=self.cat_features)
           elif estimator_name == "lgb" and self.cat_features:
               self.estimator_single.fit(X_single, y, categorical_feature=self.cat_features)
           else:
               self.estimator_single.fit(X_single, y)
       elif tp == "default":
           name_param_tp = self.get_default_params_and_name(estimator)
           estimator_name, params = name_param_tp[0], name_param_tp[1]
           if estimator_name == "catboost":
               name_estimator_dict = self.tune_parameter(estimator, "hpopt", params, X_single, y,scoring)
           else:
               name_estimator_dict = self.tune_parameter(estimator, "bayes", params, X_single, y,scoring)
           self.estimator_single = name_estimator_dict[estimator_name]
       else:
           tp_manner = tp.get("tp_manner")
           params = tp.get("params")
           name_estimator_dict = self.tune_parameter(estimator, tp_manner, params, X_single, y, scoring)
           self.estimator_single = name_estimator_dict[estimator_name]
    def get_weights(self,X,y,metric="accuracy"):
        for estimator_name, estimator in self.fited_child_learner.items():
            X_sel = self.get_X_sel(X,estimator_name)
            y_hat = estimator.predict(X_sel)
            metrics_dict = self.get_metrics(y_hat, y)
            acc = metrics_dict.get(metric)
            self.weighted_child_learner.update({estimator_name: acc})
        value_ls = self.weighted_child_learner.values()
        total = sum(value_ls)
        self.weighted_child_learner = {k: v / total for k, v in self.weighted_child_learner.items()}
        print(self.weighted_child_learner)
    # this is the function to train the model, including feature selection, parameter tuning, train the child and parent estimators
    # we use multiprocessing in the process of tuning feature and paremeter
    def fit(self,X,y,scoring="neg_log_loss",method="maxProba"):
        print("begin to fit the model")
        if self.single==None:
            if method in ["class","proba","maxProba"]:
                self.tf(X, y, scoring)
                self.tp(X, y, scoring)
                self.stacking_mode(X, y, scoring,method)
            elif method in ["average","voting","weighted_average","best",None]:
                self.tf(X, y, scoring)
                self.tp(X, y, scoring)
            else:
                print("please use the right method keyword")
                return
            self.get_weights(X,y,"f1_weighted")
        else:
            self.single_mode(X,y,scoring)

    # this is used to predict class labels
    def predict(self,X,method="maxProba"):
        print("begin to predict label")
        if self.single==None:
            # if len(self.fited_child_learner) == 1:
            #     estimator, = self.fited_child_learner.values()
            #     estimator_name, = self.fited_child_learner.keys()
            #     X_sel = self.get_X_sel(X,estimator_name)
            #     return estimator.predict(X_sel)
            if method in ["class","proba","maxProba"]:
                X_parent = self.get_X_parent(X, method=method)
                # print(X_parent.shape)
                # print(self.selector_parent)
                X_sel = self.get_X_sel_single(X_parent, self.selector_parent)
                # print(X_parent.shape)
                y_hat = self.parent_learner.predict(X_sel)
                y_hat = np.reshape(y_hat,-1)
            elif method in ["average","weighted_average"]:
                X_parent = self.get_X_parent(X, method=method)
                y_hat = np.argmax(X_parent, axis=1)
            elif method in ["voting","best"]:
                X_parent = self.get_X_parent(X, method=method)
                y_hat = X_parent
            else:
                print("please use the right method keyword")
                return
            return y_hat
        else:
            X_single = self.get_X_sel_single(X, self.selector_single)
            y_hat = self.estimator_single.predict(X_single)
            y_hat = np.reshape(y_hat, -1)
            return y_hat
    # this is used to predict class probabilitis
    # method here limited in [None,"feature_weighted","average","voting_weighted","best"]
    def predict_proba(self,X,method="maxProba"):
        print("begin to predict probability")
        if self.single==None:
            # if len(self.fited_child_learner) == 1:
            #     estimator, = self.fited_child_learner.values()
            #     estimator_name, = self.fited_child_learner.keys()
            #     X_sel = self.get_X_sel(X,estimator_name)
            #     return estimator.predict_proba(X_sel)
            if method in ["class","proba","maxProba"]:
                X_parent = self.get_X_parent(X, method=method)
                X_sel = self.get_X_sel_single(X_parent, self.selector_parent)
                return self.parent_learner.predict_proba(X_sel)
            elif method in ["average","average_average"]:
                X_parent = self.get_X_parent(X, method=method)
                return X_parent
            elif method == "best":
                best_estimator_name = max(self.weighted_child_learner, key=self.weighted_child_learner.get)
                best_estimator = self.fited_child_learner.get(best_estimator_name)
                X_sel = self.get_X_sel(X,best_estimator_name)
                y_prob = best_estimator.predict_proba(X_sel)
                return y_prob
            else:
                print("use the right method keyword")
                return
        else:
            X_single = self.get_X_sel_single(X, self.selector_single)
            return self.estimator_single.predict_proba(X_single)






