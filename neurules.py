import numpy as np
import torch
import torch.nn as nn


class Discretizing_Layer(nn.Module):
    def __init__(self, n_features, predicates_per_feature, data_limits, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.predicates_per_feature = predicates_per_feature
        self.limits = data_limits.detach().clone()

        self.cut_points = nn.Parameter(self.limits.detach().clone(), requires_grad=True)

        D = 2
        fixed_weights = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1, dtype=torch.float32), [D+1])
        # repeat per feature
        self.fixed_weights = fixed_weights.repeat(n_features,1)
        self.fixed_weights = nn.Parameter(self.fixed_weights.clone().detach(),requires_grad=False)
        self.is_discrete = [False for i in range(n_features)]

    def forward(self, x):
        cut_points = self.cut_points
        x = x.unsqueeze(2)
        lower_threshold = cut_points[:,0,:]
        upper_threshold = cut_points[:,1,:]
        

        in_interval = (2*x - lower_threshold)/self.temperature
        below_interval = (x)/self.temperature
        above_interval = (3*x - lower_threshold - upper_threshold)/self.temperature

        # Use the Log-Sum-Exp trick for numerical stability
        max_interval = torch.maximum(torch.maximum(in_interval, below_interval), above_interval).detach()
        numerator = torch.exp(in_interval - max_interval)
        denominator = (
            torch.exp(in_interval - max_interval) +
            torch.exp(below_interval - max_interval) +
            torch.exp(above_interval - max_interval)
        )
        if torch.isinf(numerator).any() or torch.isinf(denominator).any():
            print("Inf in numerator layer")
            #print(self.temperature)
            #print(numerator)
            #print(denominator)

        output = numerator / denominator

        # mid_point = (lower_threshold + upper_threshold) / 2

        # mid_point_in = (2*mid_point - lower_threshold)/self.temperature
        # mid_point_below = (mid_point)/self.temperature
        # mid_point_above = (3*mid_point - lower_threshold - upper_threshold)/self.temperature
        # # Use the Log-Sum-Exp trick for numerical stability
        # max_mid_point = torch.maximum(torch.maximum(mid_point_in, mid_point_below), mid_point_above).detach()
        # mid_point_numerator = torch.exp(mid_point_in - max_mid_point)
        # mid_point_denominator = (
        #     torch.exp(mid_point_in - max_mid_point) +
        #     torch.exp(mid_point_below - max_mid_point) +
        #     torch.exp(mid_point_above - max_mid_point)
        # )
        # mid_point_max = mid_point_numerator / mid_point_denominator
        # mid_point_max = torch.clamp(mid_point_max, min=0.5)
        # mid_point_max.unsqueeze_(0)
        # scaling_factor = 1 / (mid_point_max)

        # output = output * scaling_factor

        
        return output

    def fix_parameters(self):
        self.cut_points.data, _ = torch.sort(self.cut_points.data,dim=1)
        # span = (self.limits[:,1] - self.limits[:,0])
        
        # for i in range(self.cut_points.shape[0]):
        #     for j in range(self.cut_points.shape[2]):
        #         self.cut_points.data[i,0,j] = torch.max(self.limits[i,0]+span[i],self.cut_points.data[i,0,j])
        #         self.cut_points.data[i,1,j] = torch.min(self.limits[i,1]-span[i],self.cut_points.data[i,1,j])
        return
    
    def get_predicates(self,data_limits,feature_names=None,scaler=None,as_string=True):
        cut_points = self.cut_points.data
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(cut_points.shape[0])]
        if scaler is not None:
            all_thresholds = np.zeros(cut_points.shape)

            for i in range(self.predicates_per_feature):
                thresholds = scaler.inverse_transform(cut_points[:,:,i].detach().cpu().numpy().T).T
                all_thresholds[:,:,i] = thresholds            
            data_limits = scaler.inverse_transform(data_limits.detach().cpu().numpy().T).T
            cut_points = all_thresholds
        else:
            cut_points = cut_points.detach().cpu().numpy()
            data_limits = data_limits.detach().cpu().numpy()
        
        predicates = []
        tuple_predicates = []
        for i in range(cut_points.shape[0]):
            for j in range(self.predicates_per_feature):
                lower_bound = cut_points[i,0,j]
                upper_bound = cut_points[i,1,j]
                if lower_bound < data_limits[i,0] and upper_bound > data_limits[i,1]:
                    predicates.append("True")
                    tuple_predicates.append((lower_bound,upper_bound))
                    continue
                
                if self.is_discrete[i]:
                    lower_bound = np.ceil(lower_bound)
                    upper_bound = np.floor(upper_bound)
                    lower_bound = bool(lower_bound)
                    predicate = f"{feature_names[i]} = {lower_bound}"
                    predicates.append(predicate)
                    tuple_predicates.append((lower_bound,upper_bound))
                else:
                    lower_bound = np.max([data_limits[i,0],lower_bound])
                    upper_bound = np.min([data_limits[i,1],upper_bound])
                    predicate = f"{lower_bound:.2f} < {feature_names[i]} < {upper_bound:.2f}"
                    predicates.append(predicate)
                    tuple_predicates.append((lower_bound,upper_bound))
        if as_string:
            return predicates
        return tuple_predicates

    def init_cutpoints_subsample(self,X, fraction=0.1):
        for i in range(self.cut_points.shape[2]):
            cut_points = self.cut_points.data[:,:,i]
            # get random sample of X
            sample_indices = np.random.choice(X.shape[0], int(X.shape[0]*fraction), replace=False)
            X_sample = X[sample_indices,:]
            # get min and max for each feature
            min_values = np.min(X_sample, axis=0)
            max_values = np.max(X_sample, axis=0)
            cut_points[:,0,i] = min_values
            cut_points[:,1,i] = max_values
        

class And_Layer(nn.Module):
    def __init__(self, n_features, n_rules, epsilon=0.001):
        super().__init__()
        self.epsilon = epsilon
        self.n_rules = n_rules
        self.and_weights = nn.Parameter(torch.rand([n_rules,n_features],dtype=torch.float32), requires_grad=True)#,predicates_per_feature],dtype=torch.float32), requires_grad=True)
        
        self.and_weights.data[:] = 0.5

    
        self.relu = nn.ReLU()

    def forward(self, x):
        and_weights = self.relu(self.and_weights)
        
        # swap 1 and 2 axes of x
        
        x = x.permute(0,2,1)
        
        weight_sum = and_weights.sum(axis=1) #+ 1e-5
        eta = (self.epsilon / weight_sum)
        eta = eta.detach()
        eta = eta.unsqueeze(0)
        eta = eta.unsqueeze(2)
        # geometric weight mean

        inverse_sum = (1+eta)/(x+eta)
        
        and_weights = and_weights.unsqueeze(0)

        weighted_inverse_sum = inverse_sum*and_weights
        weighted_inverse_sum = weighted_inverse_sum.sum(dim=2)


        #inverse_sum = inverse_sum.reshape([x.shape[0],-1])
        #weighted_inverse_sum = torch.multiply(inverse_sum,and_weights)
        #weighted_inverse_sum = weighted_inverse_sum.reshape([x.shape[0],self.n_rules,-1])
        #weighted_inverse_sum = torch.sum(weighted_inverse_sum,dim=[2])

        res = weight_sum/(weighted_inverse_sum)
        # if torch.isnan(weight_sum).any():
        #     print("NaNs in weight_sum")
        # if torch.isnan(weighted_inverse_sum).any():
        #     print("NaNs in weighted_inverse_sum")
        # if torch.isnan(res).any():
        #     print("NaNs in res")
        return res

    def fix_parameters(self):
        #self.and_weights.data = torch.clamp(self.and_weights.data,max=5)
        pass

class NeuralRuleList(nn.Module):
    def __init__(self, config, limits):
        super().__init__()
        self.n_rules = config.n_rules
        self.n_classes = config.n_classes

        self.discretizer = Discretizing_Layer(config.n_features, config.n_rules, limits, config.predicate_temperature)
        if not config.random_cutpoints:
            # randomly scale cut points
            pass
            self.discretizer.cut_points.data = limits.unsqueeze(2).repeat(1,1,self.n_rules)
        self.rules = And_Layer(config.n_features,config.n_rules)
        self.rule_order = nn.Parameter(torch.rand([config.n_rules],dtype=torch.float32),requires_grad=True)
        self.rule_weights = nn.Linear(config.n_rules,config.n_classes,bias=False,dtype=torch.float32)

        # if config.init == "normal":
        #     self.rule_weights.weight.data = torch.rand([config.n_classes,config.n_rules],dtype=torch.float32)
        # elif config.init == "xavier": 
        #     nn.init.xavier_uniform_(self.rule_weights.weight)
        # elif config.init == "const":
        #     self.rule_weights.weight.data.fill_(0.5)
        # part = self.n_rules//self.n_classes

        # if self.n_rules % self.n_classes == 0:
        #     for i in range(config.n_classes):
        #         self.rule_weights.weight.data[i,i*part:(i+1)*part] = 10
        # else:
        #     for i in range(config.n_classes):
        #         if i==config.n_classes -1:
        #             self.rule_weights.weight.data[i,i*part:(i+1)*part + 1] = 10
        #         else:
        #             self.rule_weights.weight.data[i,i*part:(i+1)*part] = 10
        #     #self.rule_weights.weight.data[:,part]=2

        self.selector_temperature = config.selector_temperature

        

    def forward(self, x, return_rules=False):

        predicates = self.discretizer(x)

        individual_rules = self.rules(predicates)
        #print(individual_rules)
        # ensure that self.rule_order is positive
        rule_order = self.rule_order - torch.min(self.rule_order).item() + 1
        
        if torch.isnan(rule_order).any():
            print("NaNs in rule_order")

        rule_weight = individual_rules * rule_order
        # use gumbel softmax to take active rule with max weight
        if True:
            active_rule = nn.functional.gumbel_softmax(rule_weight,tau=self.selector_temperature,hard=False)
            #torch.set_printoptions(sci_mode=False)
            #print("Active rule")
            #print(active_rule[0])
        else:
            # take arg max
            b = torch.argmax(rule_weight,dim=1)
            active_rule = torch.nn.functional.one_hot(b,num_classes=rule_weight.shape[1]).float()
        #active_rule = nn.functional.softmax(rule_weight/self.selector_temperature,dim=1)
        prediction = self.rule_weights(active_rule)
        if return_rules:
            return prediction, individual_rules, active_rule
        
        return prediction
    
    def get_rule_selection(self,x):
        predicates = self.discretizer(x)
        rule_results = self.rules(predicates)
        rule_order = self.rule_order - torch.min(self.rule_order).item() + 1
        rule_weight = rule_results * rule_order
        if True:
            active_rule = nn.functional.gumbel_softmax(rule_weight,tau=self.selector_temperature,hard=False)
        else:
            # take arg max
            # turn into one hot
            b = torch.argmax(rule_weight,dim=1)
            active_rule = torch.nn.functional.one_hot(b,num_classes=rule_weight.shape[1]).float()
        return active_rule.detach()
    
    def get_weighted_priority(self,x):
        predicates = self.discretizer(x)
        rule_results = self.rules(predicates)
        
        rule_order = self.rule_order - torch.min(self.rule_order).item() + 1
        rule_weight = rule_results * rule_order
        return rule_weight.detach()
    
    def get_rule_activations(self,x):
        predicates = self.discretizer(x)
        rule_results = self.rules(predicates)
        return rule_results.detach()
    
    def get_active_rules(self,x, threshold=0.5):
        predicates = self.discretizer(x)
        rule_results = self.rules(predicates).detach()
        is_active = rule_results > threshold
        is_active = is_active.squeeze().detach().cpu().numpy()
        

        indices = np.where(is_active == True)[0]
        return indices


    def get_probabilities_per_rule(self):
        probs = nn.functional.softmax(self.rule_weights.weight,dim=0)
        return probs.detach()

    def fix_parameters(self):
        self.discretizer.fix_parameters()
        self.rules.fix_parameters()
        return
    
    def reduce_predicate_temperature(self, factor):
        self.discretizer.temperature = self.discretizer.temperature/factor
        return

    def reduce_selector_temperature(self, factor):
        self.rule_selector.temperature = self.rule_selector.temperature/factor
        return
    
    
    

    def get_rule(self,index, data_limits, threshold=0., scaler_x=None, feature_names=None, scaler_y=None):
        prediction = (self.rule_weights.weight[0,index]).detach() 
        if scaler_y is not None:
            prediction = prediction* scaler_y.scale_

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(data_limits.shape[0])]

        cut_points = self.discretizer.cut_points
        if scaler_x is not None:
            all_thresholds = np.zeros(cut_points.shape)

            for i in range(self.discretizer.predicates_per_feature):
                thresholds = scaler_x.inverse_transform(cut_points[:,:,i].detach().cpu().numpy().T).T
                all_thresholds[:,:,i] = thresholds            
            data_limits = scaler_x.inverse_transform(data_limits.detach().cpu().numpy().T).T
            cut_points = all_thresholds
        else:
            cut_points = cut_points.detach().cpu().numpy()
            data_limits = data_limits.detach().cpu().numpy()
        rule = []
        and_weights = self.rules.and_weights.data[index,:].detach().cpu().numpy()

        for i in range(and_weights.shape[0]):
            lower_bound = data_limits[i,0]
            upper_bound = data_limits[i,1]
            feature_weight = and_weights[i]
            if feature_weight <= threshold:
                continue
            
            lower_threshold = cut_points.data[i,0,index]
            upper_threshold = cut_points.data[i,1,index]
            if self.discretizer.is_discrete[i]:
                lower_threshold = np.ceil(lower_threshold)
                upper_threshold = np.floor(upper_threshold)
                lower_threshold = bool(lower_threshold)
                lower_bound = np.max([lower_bound,lower_threshold])
                
            else:
                lower_bound = np.max([lower_bound,lower_threshold])
                upper_bound = np.min([upper_bound,upper_threshold])
            if data_limits[i,0] == lower_bound and data_limits[i,1] == upper_bound:
                continue
            if self.discretizer.is_discrete[i]:
                predicate = f"{feature_names[i]} = {lower_bound}"
                rule.append(predicate)
            else:
                predicate = ""
                if lower_bound > data_limits[i,0]:
                    predicate = predicate + f"{lower_bound:.2f} < "
                predicate = predicate + f"{feature_names[i]}"
                if upper_bound < data_limits[i,1]:
                    predicate = predicate + f" < {upper_bound:.2f}"
                rule.append(predicate)
        rule = " ∧ ".join(rule)
        return rule, prediction
    
    def get_top_k_rules(self, x, k):
        rule_activations = self.get_rule_selection(x)
        top_activations, top_indices = torch.topk(rule_activations,k)
        top_activations = top_activations.squeeze().detach().cpu().numpy().tolist()
        top_indices = top_indices.squeeze().detach().cpu().numpy().tolist()
        return top_activations, top_indices

    def print_rule_list(self, limits, scaler_x=None, feature_names=None, scaler_y=None):
        order = self.rule_order.detach().cpu().numpy() - np.min(self.rule_order.detach().cpu().numpy()) + 1
        indices = np.argsort(order)[::-1]
        description = ""
        for i in indices:
            prob = nn.functional.softmax(self.rule_weights.weight[:,i],dim=0).detach().cpu().numpy()*100
            # log prob
            # class by argmax
            pred = np.argmax(prob)
            prob = prob[pred]
            
            if i == self.n_rules:
                description += f"else for class {pred} with likelihood {prob:.2f}%\n"
                break

            rule,_ = self.get_rule(i,limits,scaler_x=scaler_x,feature_names=feature_names,scaler_y=scaler_y,threshold=0.)
            if rule == "":
                continue
            priority = order[i]
            description += f"Rule {i} with {priority:.2f} for class {pred} with likelihood {prob:.2f}%: {rule}\n"
        return description

    def to_hard_rule_list(self):
        model = HardRuleList(self.discretizer.cut_points.data.detach().clone(),self.rules.and_weights.data.detach().clone(),self.rule_order.detach().clone(),self.rule_weights.weight.detach().clone().T)
        return model

class HardRuleList(nn.Module):
    def __init__(self, cut_points, and_weights, rule_order, rule_weights, path=None):
        super().__init__()
        if path:
            self.load_model_variables(path)
        else:
            self.cut_points = cut_points
            self.and_weights = and_weights
            self.rule_order = rule_order
            self.rule_weights = rule_weights
            self.index_order = torch.argsort(rule_order,descending=True)
        
    @classmethod
    def from_file(cls, path):
            return cls(None, None, None, None, path=path)
    
    def forward(self, x, ret_covered_by=False):
        pred = torch.zeros((x.shape[0],self.rule_weights.shape[1]),dtype=torch.float32)
        covered_by = np.zeros(x.shape[0],dtype=int) - 1
        for s in range(x.shape[0]):
            sample = x[s,:]
            for i in self.index_order:
                lower_bound = self.cut_points[:,0,i]
                upper_bound = self.cut_points[:,1,i]
                predicate = (sample > lower_bound) & (sample < upper_bound)
                and_weights = self.and_weights[i,:] <= 0.
                if and_weights.all():
                    continue
                rule = (predicate | and_weights).all()
                if rule:
                    pred[s,:] = self.rule_weights[i,:]
                    covered_by[s] = i
                    break
        if ret_covered_by:
            return pred, covered_by
        return pred
    
    def load_model_variables(self,filepath):
        checkpoint = torch.load(filepath)
        self.cut_points = checkpoint['cut_points']
        self.and_weights = checkpoint['and_weights']
        self.rule_order = checkpoint['rule_order']
        self.rule_weights = checkpoint['rule_weights']
        self.index_order = torch.argsort(self.rule_order, descending=True)
        return self
    
    def save(self,path):
        torch.save({
            'cut_points': self.cut_points,
            'and_weights': self.and_weights,
            'rule_order': self.rule_order,
            'rule_weights': self.rule_weights
        }, path)
        return