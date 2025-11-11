import torch

class Train_Config:
    def __init__(self, seed=42, n_epochs=2000, lr=0.002, batch_size=-1, shuffle=True, min_support=0.02, max_support=1., lambd=1., device=None):
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_support = min_support
        self.max_support = max_support
        self.lambd = lambd
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

class Rule_Config():
    def __init__(self,n_features, n_classes, n_rules = 10, predicate_temperature = 0.2, selector_temperature=0.5, 
                 random_cutpoints=False, temp_reduction_selector=5,temp_reduction_predicate=5,init="normal"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_rules = n_rules
        self.predicate_temperature = predicate_temperature
        self.selector_temperature = selector_temperature
        self.random_cutpoints = random_cutpoints
        self.init = init

        self.temp_reduction_selector = temp_reduction_selector
        self.temp_reduction_predicate = temp_reduction_predicate
        self.schedule_selector_temperature = {"start":self.selector_temperature,
                                              "end":self.selector_temperature/self.temp_reduction_selector,"progress":"linear"}
        self.schedule_predicate_temperature = {"start":self.predicate_temperature,
                                                  "end":self.predicate_temperature/self.temp_reduction_predicate,"progress":"linear"}