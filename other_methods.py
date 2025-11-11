from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
from imodels import RuleFitClassifier, RuleFitRegressor, GreedyRuleListClassifier, OptimalRuleListClassifier, BayesianRuleListClassifier, BoostedRulesClassifier, BoostedRulesRegressor,C45TreeClassifier
from mdl_rulelist import RuleListClassifier, RuleListRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
#from sklweka.classifiers import WekaEstimator

try:
    from corels import CorelsClassifier
except:
    "Setz halt dein environment richtig auf!"

def fit(self, X, y, feature_names=None, prediction_name="prediction"):
        """
        Build a CORELS classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8.
        y : array-line, shape = [n_samples]
            The target values for the training input. Must be binary.

        feature_names : list, optional(default=None)
            A list of strings of length n_features. Specifies the names of each
            of the features. If an empty list is provided, the feature names
            are set to the default of ["feature1", "feature2"... ].
        prediction_name : string, optional(default="prediction")
            The name of the feature that is being predicted.
        Returns
        -------
        self : obj
        """
        print("fitting")
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        elif feature_names is None:
            feature_names = ['X_' + str(i) for i in range(X.shape[1])]

        X = X.astype(int)
        np.random.seed(self.random_state)
        feature_names = feature_names#.tolist()
        print(feature_names)
        CorelsClassifier.fit(self, X, y, features=feature_names, prediction_name=prediction_name)
        # try:
        self._traverse_rule(X, y, feature_names)
        # except:
        #     self.str_print = None
        self.complexity_ = self._get_complexity()
        return self

OptimalRuleListClassifier.fit = fit

parameter_name_mapper = {
    "rulefit": {
        "n_rules": "max_rules",
    },
    "greedy_rule_list": {
        "n_rules": "max_depth",
    },
    "optimal_rule_list": {
        "n_rules": "max_depth",
    },
    "bayesian_rule_list": {
        "n_rules": "listlengthprior",
    },
    "boosted_rules": {
        "n_rules": "n_estimators",
    },
    "xgboost": {
        "n_rules": "n_estimators",
    },
    "mdl-rule-list": {
        "n_rules": "max_rules",
    },
    "random_forest": {
        "n_rules": "n_estimators",
    },
    "mlp": {},
    "CART": {
        "n_rules": "max_leaf_nodes",
    },
    "C45": {
        "n_rules": "max_rules",
    },
    "ripper": {
        "n_rules": "max_rules",
    }
}

def get_model(name, classification, params,feature_names, args):
    parameter_mapper = {}
    renamer = parameter_name_mapper[name]
    for key, value in params.items():
        if key in renamer:
            parameter_mapper[renamer[key]] = value
    params = parameter_mapper
    if name == "rulefit":
        if classification:
            return RuleFitClassifier(**params)
        else:
            return RuleFitRegressor(**params)
    elif name == "greedy_rule_list":
        if classification:
            return GreedyRuleListClassifier(max_depth=args.max_depth)
    elif name == "optimal_rule_list":
        if classification:
            if not hasattr(np, 'bool'):
                np.bool = np.bool_   
            return OptimalRuleListClassifier(max_card=args.max_card,
                                              c=args.c, n_iter=args.n_iter,
                                                min_support=args.corels_min_support,verbosity=["progress"])
    elif name == "boosted_rules":
        if classification:
            return BoostedRulesClassifier(**params)
        else:
            return BoostedRulesRegressor(**params)
    elif name == "mdl-rule-list":
        if classification:
            return RuleListClassifier(max_depth=3, beam_width=200, n_cutpoints=5, max_rules=args.n_rules)
        else:
            return RuleListRegressor(**params)
    elif name == "xgboost":
        if classification:
            return XGBClassifier(n_estimators=args.xg_n_estimators, max_depth=args.xg_max_depth, learning_rate=args.xg_learning_rate,reg_lambda=args.xg_reg_lambda)
        else:
            return HistGradientBoostingRegressor(**params)
    elif name == "random_forest":
        if classification:
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)
    elif name == "mlp":
        if classification:
            return MLPClassifier(**params)
        else:
            return MLPRegressor(**params)
    elif name == "CART":
        if classification:
            return DecisionTreeClassifier(**params)
        else:
            return DecisionTreeRegressor(**params)
    elif name == "C45":
        if classification:
            return C45TreeClassifier(**params)
    elif name == "ripper":
        Ripper = WekaEstimator(classname="weka.classifiers.rules.JRip")
        return Ripper

        
def get_rule(model, name):
    if name == "mdl-rule-list:":
        return str(model)
    elif name == "greedy_rule_list":
        return str(model)
    else:
        return str(model)