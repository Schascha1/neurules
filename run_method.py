import argparse

from neurules import *
from other_methods import *
from configs import *
from utils import *
from training import *
from data_loaders import *


import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import time
import rlnet.networkTorch_multiClass as rulelistnet
from rrl import models as rrl_models
import drs.DRNet as drnet 

import warnings



if __name__ == "__main__":

    warnings.filterwarnings("ignore")


    torch.manual_seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    # parse dataset and method
    supported_methods = ["rulefit", "greedy_rule_list", "optimal_rule_list", "bayesian_rule_list", "CART","mdl-rule-list",
                         "boosted_rules", "xgboost","our-rule-list","our-rule-set","rlnet","rrl","drs"]
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--outpath", type=str)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--device",type=str,default="cpu")

    # parse hyperparameters
    parser.add_argument("--n_rules", type=int, default=-1)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--min_support", type=float, default=0.2)
    parser.add_argument("--max_support", type=float, default=0.9)
    parser.add_argument("--lambd", type=float, default=0.4)
    parser.add_argument("--init", type=str, default="const")
    parser.add_argument("--predicate_temperature", type=float, default=0.2)
    parser.add_argument("--selector_temperature", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="linear")

    parser.add_argument('--rl_net_learning_rate', type=float, default=1e-2,help='The learning rate of the training')
    parser.add_argument('--rl_net_lambda_and', type=float, default=1e-3,  help='The scaling factor for the AND layer regularization')
    parser.add_argument('--rl_net_epochs', type=int, default=3000,  help='The number of training epochs')
    parser.add_argument('--rl_net_limit', type=int, default=1000,  help='The number of epochs the balanced loss should be used')
    parser.add_argument('--rl_net_l2_lambda', type=float, default=0.001,  help='The scaling factor of the L2 regularization')


    parser.add_argument('--drs_epochs', type=int, default=3000, help='The number of training epochs for DRS')
    parser.add_argument('--drs_batch_size', type=int, default=2000, help='The training batch size for DRS')
    parser.add_argument('--drs_lr', type=float, default=0.01, help='The learning rate for DRS')
    parser.add_argument('--drs_and_lam', type=float, default=0.0001, help='The AND layer regularization factor for DRS')
    parser.add_argument('--drs_or_lam', type=float, default=0.0001, help='The OR layer regularization factor for DRS')


    parser.add_argument('--listlengthprior', type=int, default=10, help='The prior for the list length')
    parser.add_argument('--listwidthprior', type=int, default=2, help='The prior for the list width')
    parser.add_argument('--maxcardinality', type=int, default=3, help='The maximum cardinality')
    parser.add_argument('--minsupport', type=float, default=0.05, help='The minimum support')
    parser.add_argument('--n_chains', type=int, default=10, help='The number of chains')


    parser.add_argument('--c', type=float, default=0.01, help='Regularization parameter')
    parser.add_argument('--n_iter', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--max_card', type=int, default=5, help='Maximum cardinality')
    parser.add_argument('--corels_min_support', type=float, default=0.01, help='Minimum support')

    parser.add_argument("--max_depth", type=int, default=5)

        
    parser.add_argument('--beam_width', type=int, default=100, help='Beam width for the search')
    parser.add_argument('--n_cutpoints', type=int, default=5, help='Number of cutpoints for discretization')

    parser.add_argument('--xg_learning_rate', type=float, default=0.1, help='Learning rate for the model')
    parser.add_argument('--xg_max_depth', type=int, default=7, help='Maximum depth of the tree')
    parser.add_argument('--xg_n_estimators', type=int, default=300, help='Number of estimators')
    parser.add_argument('--xg_reg_lambda', type=float, default=0, help='Regularization parameter lambda')

    parser.add_argument("--ablation_thresholding", type=bool, default=False)
    parser.add_argument("--ablation_thresholding_kind", type=str, default="kmeans")
    parser.add_argument("--ablation_order", type=bool, default=False)
    parser.add_argument("--ablation_conjunction", type=bool, default=False)
    parser.add_argument("--ablation_contin", action='store_true')

    parser.add_argument("--ablation_cuts", action='store_true')
    parser.add_argument("--ablation_bins", type=int, default=5)




    #recover the arguments
    args = parser.parse_args()
    dataset = args.dataset
    method = args.method
    n_rules = args.n_rules
    outpath = args.outpath
        

    train_config = Train_Config(
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        min_support=args.min_support,
        max_support=args.max_support,
        lambd=args.lambd
    )
    train_config.device = args.device
    params_dict = {"n_rules":args.n_rules} if args.n_rules != -1 else {}
    
    classification = not (dataset in regression_datasets)
    try:
        dataset_object, _ = get_dataset(dataset)
    except:
        print(f"Dataset {dataset} not found locally. Please download it from the cited source.")
        exit(1)


    X = dataset_object["data"]
    Y = dataset_object["target"]


    
    scaler_X = StandardScaler()
    
    X, dataset_object["feature_names"] = binarize_categorical(X, dataset_object["feature_names"])
    # ablation thresholding
    # discretize ablation
    if args.ablation_thresholding:
        n_bins = 2
        if args.ablation_thresholding_kind == "kmeans":
            X = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit_transform(X)
        elif args.ablation_thresholding_kind == "quantile":
            X = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit_transform(X)
        else:
            X = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(X)
        dataset_object["feature_names"] = [f"{f}_{i}" for f in dataset_object["feature_names"] for i in range(n_bins)]

    if method == "optimal_rule_list" and not args.ablation_cuts:
        n_bins = 5
    else:
        n_bins = args.ablation_bins
    if  method=="optimal_rule_list" or method == "bayesian_rule_list":# or method == "drs" :#, "Only available for Corels and SBRL"
        print(method,n_bins,args.ablation_bins)
        #df = pd.DataFrame(X,columns=dataset_object["feature_names"])
        print("before: ",X.shape)
        disc = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
        X = disc.fit_transform(X)
        print("after: ",X.shape)
        dataset_object["feature_names"] = disc.get_feature_names_out()
        print(np.unique(X))
        

    
    if method!="optimal_rule_list" and method!="bayesian_rule_list":
        X = scaler_X.fit_transform(X)
        
    is_discrete = []
    for i in range(X.shape[1]):
        is_discrete.append(len(np.unique(X[:, i])) < 5)
    if method == "our-rule-list" or method == "rlnet" or method == "rrl" or method == "drs":
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).long()
    if not classification:
        scaler_Y = StandardScaler()
        Y = scaler_Y.fit_transform(Y.reshape(-1, 1)).reshape(-1)
    # 5 fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    accuracies = []
    roc_auc = []
    runtimes = []

    k = 1
    n_classes = len(np.unique(Y)) if classification else 1
    n_rules = args.n_rules if n_classes == 2 else n_classes*args.n_rules//2

    model_config = Rule_Config(X.shape[1],n_classes,predicate_temperature=args.predicate_temperature,
                               selector_temperature=args.selector_temperature,init=args.init) 

    for train_index, test_index in kfold.split(X):
        t = time.time()
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        
        rule = ""

        if method == "our-rule-list":
            if n_rules != -1:
                model_config.n_rules = n_rules
            limits = get_data_limits(X_train)
            if method == "our-rule-list":
                model = NeuralRuleList(model_config, limits)
            if args.ablation_conjunction:
                model.rules.epsilon = 0
            if args.ablation_order:
                model.rule_order.requires_grad = False


            model.discretizer.is_discrete = is_discrete

            selector_schedule = None
            predicate_schedule = Temperature_Scheduler(train_config.n_epochs//2,model_config.schedule_predicate_temperature)
            if method == "our-rule-list":
                selector_schedule = Temperature_Scheduler(train_config.n_epochs//2,model_config.schedule_selector_temperature)

            loss = torch.nn.CrossEntropyLoss() if classification else torch.nn.MSELoss()

            pre_cut = model.discretizer.cut_points.clone()
            model = train_model(model, X_train, Y_train, loss,train_config,temperature_predicate_schedule=predicate_schedule,temperature_selector_schedule=selector_schedule,return_losses=False,disable_tqdm=True)
            model.eval()
            hard_rl = model.to_hard_rule_list()
            Y_score = hard_rl(X_test).detach()
            
            Y_pred = Y_score.argmax(dim=1) if classification else Y_score
            Y_pred = Y_pred.cpu().numpy()
            Y_score = Y_score.cpu().numpy()[:,1]

            Y_soft = model(X_test).detach()
            Y_soft = Y_soft.argmax(dim=1) if classification else Y_soft
            Y_soft = Y_soft.cpu().numpy()
            difference = np.mean(Y_soft != Y_pred)


            rule = model.print_rule_list(limits,scaler_x=scaler_X,feature_names=dataset_object["feature_names"])
        elif method == "rlnet":

            if n_rules == -1:
                n_rules = model_config.n_rules
            model = rulelistnet.Network(X_train.shape[1],n_rules,n_classes)
            # split train set 80 20
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
            rulelistnet.train(model,X_train, Y_train,X_val,Y_val,round(0.05*X_train.shape[0]),n_classes,learning_rate=args.rl_net_learning_rate,lambda_and=args.rl_net_lambda_and,epochs=args.rl_net_epochs,l2_lambda=args.rl_net_l2_lambda)
            model.eval()
            Y_score,_ = model(X_test)
            Y_score = Y_score.detach()
            Y_pred = Y_score.argmax(dim=1) if classification else Y_score
            Y_pred = Y_pred.cpu().numpy()
            Y_score = Y_score.cpu().numpy()[:,1]
            rule = rulelistnet.print_rules(model)

        elif method == "rrl":
            #dimlist = 
            model = rrl_models.Net()

            pass

        elif method == "drs":
            if n_rules == -1:
                n_rules = model_config.n_rules
            model = drnet.DRNet(X_train.shape[1],n_rules,1)
            # make torch dataset
            trainset = torch.utils.data.TensorDataset(X_train,Y_train.float())
            testset = torch.utils.data.TensorDataset(X_test,Y_test.float())
            drnet.train(model,trainset,testset,device="cpu",lr=args.drs_lr,and_lam=args.drs_and_lam,or_lam=args.drs_or_lam,
                        epochs=args.drs_epochs,batch_size=args.drs_batch_size)
            model.eval()
            Y_score = model(X_test).detach()
            Y_pred = Y_score >=0 if classification else Y_score
            Y_pred = Y_pred.cpu().numpy()
            rule = model.get_rules(header=dataset_object["feature_names"])
            rules = []
            for r in rule:
                rules.append(" AND ".join(r))
            rule = "\n".join(rules)

        elif method == "bayesian_rule_list":
            model = get_model(method, classification, params_dict,dataset_object["feature_names"],args)
            model = BayesianRuleListClassifier(listlengthprior=args.n_rules/3,
                                                listwidthprior=args.listwidthprior,
                                                maxcardinality=args.maxcardinality,
                                                 minsupport=args.minsupport,
                                                 n_chains=args.n_chains,verbose=True)
            # turn into df 
            
            
            X_train = pd.DataFrame(X_train, columns=dataset_object["feature_names"])
            X_test = pd.DataFrame(X_test, columns=dataset_object["feature_names"])
            #X_train["target"] = Y_train.tolist()
            #X_test["target"] = Y_test.tolist()

            #disc = MDLPDiscretizer(X_train,"target", cuts=args.ablation_bins)
            #disc.to_return = disc.to_return.drop(columns=["target"])
            model.fit(X_train, Y_train)
            model._setlabels(X_train)
            #disc2 = MDLPDiscretizer(X_test,"target",boundaries=disc._boundaries, cuts=disc._cuts)
            #disc2.to_return = disc2.to_return.drop(columns=["target"])

            Y_pred = np.argmax(model.predict_proba(X_test),axis=1) #model.predict(X_test)
            Y_score = np.argmax(model.predict_proba(X_test),axis=1) if classification else Y_pred
            rule = get_rule(model, method)
        
        elif method == "optimal_rule_list":
            model = get_model(method, classification, params_dict,list(dataset_object["feature_names"]),args)
            X_train = pd.DataFrame(X_train, columns=dataset_object["feature_names"])
            X_test = pd.DataFrame(X_test, columns=dataset_object["feature_names"])
            X_train = X_train#.astype(int)
            #print(X_train)
            model.fit(X_train, Y_train)
            Y_pred = np.argmax(model.predict_proba(X_test),axis=1) #model.predict(X_test)
            Y_score = model.predict_proba(X_test)[:,1] if classification else Y_pred

            rule = get_rule(model, method)
        elif method == "ripper":
            from sklweka.dataset import to_nominal_labels
            import sklweka.jvm as jvm
            from sklweka.classifiers import WekaEstimator
            jvm.start()
            min_weight = (1/(n_rules*2))*len(Y_train)
            Ripper = WekaEstimator(classname="weka.classifiers.rules.JRip",options=["-N",str(min_weight)])

            Y_train = to_nominal_labels(Y_train)

            Ripper.fit(X_train,Y_train)
            ripper_rules = str(Ripper)
            Y_pred = Ripper.predict(X_test)
            # remove _ and turn to number
            Y_pred = np.array([int(y[1:]) for y in Y_pred])
            Y_score = Y_pred
            rule = ripper_rules
        else:
            model = get_model(method, classification, params_dict,dataset_object["feature_names"],args)
            # turn into df 
            X_train = pd.DataFrame(scaler_X.inverse_transform(X_train), columns=dataset_object["feature_names"])
            X_test = pd.DataFrame(scaler_X.inverse_transform(X_test), columns=dataset_object["feature_names"])
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            Y_score = model.predict_proba(X_test)[:,1] if classification else Y_pred

            rule = get_rule(model, method)

        t = time.time() - t
        runtimes.append(t)
        if classification:
            Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test
            f1_scores.append(f1_score(Y_test, Y_pred, average="weighted"))
            accuracies.append(accuracy_score(Y_test, Y_pred))
            roc_auc.append(0)
        if os.path.exists(f"{outpath}rules/{method}/") == False:
            os.makedirs(f"{outpath}rules/{method}/")
            
        f = open(f"{outpath}rules/{method}/{dataset}_run{k}.txt","w")
        f.write(rule)
        f.close()
        k += 1
    if classification:
        f1 = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        acc = np.mean(accuracies)
        acc_std = np.std(accuracies)
        roc = np.mean(roc_auc)
        roc_std = np.std(roc_auc)
        runtime = np.mean(runtimes)
        runtime_std = np.std(runtimes)
        if args.verbose:
            print(f"Dataset: {dataset}, Method: {method}, F1: {f1}, Acc: {acc}, ROC: {roc}, Runtime: {runtime}")
        print(f"{outpath}{method}.csv")
        f = open(f"{outpath}{method}.csv", "a")
        f.write(f"{dataset};{n_rules};{f1};{f1_std};{acc};{acc_std};{roc};{roc_std};{runtime};{runtime_std}\n")
        f.close()

