# Stacking 
## this is a stacking classifier with built-in parameter-tuning and feature selection featues 
1. this classifier is built based on the stacking of several popular boosting methods (also support not boosting methods):
        included base learner "child_learner": [adaboost, gbm, xgb, lgb, catboost,svc,lr,rf,et,mlp]
2. the outputs (predictions of probabilities) of these base learner are input features of one parent learner:
        included parent learner "parent_learner": one of [adaboost, gbm, xgb, lgb, catboost,svc,lr,rf,et,mlp]
3. *  we support different stacking method by setting the "method" parameter in fit(), predict() or predict_prob():
   *  support using class as predictions by setting it as "class",
   *  support using probabilities as predictions by setting it as "proba",
   *  support using the max probability as prediction by setting it as "maxProba",
   *  support averaging method by setting it as "average",
   *  support simple voting method by setting it as "voting",
   *  support weighted voting method by setting it as "weighted_average", weights are assigned according to the prediction accuracy
   *  support using the best base learner by setting it as "best"
   *  we also support stacking with the raw features by setting  "with_raw_features=True",
4. *  we use both wrapping and embedding method to select important features,
   *  the  function is built upon RFECV() and SelectFromModel() in scikit-learn,
   *  RFECV() is a wrapping method, while SelectFromModel() is a embedding method :
        *  tune_feature = None|"default"|[{"estimator": estimator,"tf_manner": None or "wrapping" or "embedding"},...]
        *  None: do not perform feature selection
        *  "default": perform feature selection for all the base learners specified in child_learner,
        *  [{"estimator": estimator,"tf_manner": None or "wrapping" or "embedding"},...]:
            *  perform feature selection for the specified estimator with specified feature tuning manner
            *  "wrapping" means using RFECV(), while "embedding" means using SelectFromModel()
5. *  the parameter tuning function is built upon GridSearchV, RandomlizedSearchCV, BayseSearchCV, and hyperopt module:
        *  child_optimizer: None|"default"|[{"estimator": AdaBoostClassifier(), "params":params, "tp_manner":"gs"},...]
        *  parent_optimizer = None|"default"|{"params":params, "tp_manner":"gs","tf_manner":"wrapping"}
        *  None: do not perform parameter tuning
        *  "default": perform parameter tuning for every base learner specified in child_learner,
        *  [{"estimator": AdaBoostClassifier(), "params":params, "tp_manner":"gs"},...]:
                    *  a list of dict (a dict for parent_optimizer) contains
                    *  "estimator": estimator names, include adaboost, gbm, xgb, lgb, catboost,rf,svc,lr
                    *  "params": parameter search space, a dict, with parameter names as keys and search spaces as values
                    *  "tp_manner": search manners, include gs, bayes, hpopt, random
                    *  "tf_manner": only available for parent_optimizer, include values [None, "default","wrapping","embedding"]
        *  if you do not want to perform parameter tuning, just leave the child_optimizer or parent_optimizer as None,the default setting
        *  if you want to use the default tuning, set "default" to them,
        *  if you want costomized tuning, you need to set them as described above
        *  remember that the tuning process can be long, and the "parent_optimizer" is a dict instead of a list of dict
6. we also support single model pattern with "single" parameter: 
    *  it's default is None which means stacking pattern, otherwise, set it as a dict with three items
    *  single["estimator"]: one of [rf,adaboost, gbm, xgb, lgb, catboost,svc,lr]
    *  single["tf"]: one of [None,"default","wrapping","embedding"], None means no feature selection,
    *  single["tp"]: one of [None,"default",{"params":params space,"tp_manner": one of "bayes,hpopt,random,gs"}], 
                    *  None means no parameter tuning, 
                    *  "default" means default tuning, 
                    *  customized tuning is supported by setting a dict with parameter space and tuning manner
7. lightgbm and catboost support categorical features with categorical column indices,
*  when dealing with imbalanced problem, it's better to set the "class_weight" parameter as "balanced" if the learner has one
*  estimators with "class_weight" parameter:
                      *  ExtraTreesClassifier(), SVC(),LogisticRegression(),RandomForestClassifier(),LGBMClassifier()
8. For multiclass problem, you need to set the "loss_function" parameter in CatBoostClassifier() as "Multiclass" if you include it
9. We also support SVC(), LogisticRegression(),MLPClassifier() and other algorithms might be enclosed in the future
10. The reason we use hyperopt for catboost is that we couldn't fix the bug when using it with bayes optimizer
    and some research shows that hyperopt is faster than scikit-optimizer, but the latter contains more optimizing algorithms
11. The scoring function can be tricky, "roc_auc" is not supported for multiclass problem and sometimes "neg_log_loss" (this is why it's tricky),especially when your data is small and some labels are not predicted
12. Non-linear SVC() and MLP can't not be used with feature selection, cause their do not expose coef_ or feature_importantance

