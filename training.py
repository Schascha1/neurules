import torch
from torch.optim import Adam,SGD
from tqdm import tqdm
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from utils import *
import itertools
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(model, X, Y, criterion,config, temperature_selector_schedule=None, temperature_predicate_schedule=None, return_losses=False,X_test=None,Y_test=None,disable_tqdm=False):
    model.to(config.device)    
    optimizer = Adam(model.parameters(), lr=config.lr)
    model.train()
    X = X.to(config.device)
    Y = Y.to(config.device)
    losses = []
    
    batch_size = config.batch_size
    if batch_size == -1:
        batch_size = len(X)
    loader = get_data_loader(X,Y,batch_size,config.shuffle)

    all_rule_pairs = []
    for i in range(model.n_rules):
        for j in range(i+1,model.n_rules):
            all_rule_pairs.append((i,j))
    ind1,ind2 = zip(*all_rule_pairs)

    for epoch in tqdm(range(config.n_epochs), disable=disable_tqdm):
        for X, Y in loader:        
            optimizer.zero_grad()
            Y_pred, rule_activations, combined_activations = model(X,return_rules=True)
            
            fit_loss = criterion(Y_pred,Y)

            selection = (combined_activations).mean(dim=0)
            selection_loss = torch.relu(config.min_support-selection).mean()

            loss = fit_loss + selection_loss*config.lambd 
        
            loss.backward()


            if epoch < config.n_epochs//2:
                model.rule_weights.weight.grad = None
            

            if epoch % 100 == 0 and not disable_tqdm:
                torch.set_printoptions(precision=2)

            if torch.isnan(loss) and model.rules.epsilon > 0:
                print("Loss is NaN, skipping optimizer step")
                continue

            optimizer.step()
            
            optimizer.step()
            model.fix_parameters()
            if X_test is not None and Y_test is not None:
                test_loss = eval_model(model, X_test, Y_test, F1_Loss(), config)
                losses.append(test_loss.item())
            else:
                losses.append(loss.item())
            
        
            if temperature_selector_schedule is not None and epoch >= config.n_epochs//2:
                model.selector_temperature = temperature_selector_schedule.get_temperature()
            if temperature_predicate_schedule is not None and epoch >= config.n_epochs//2:
                model.discretizer.temperature = temperature_predicate_schedule.get_temperature()


    if return_losses:
        return model, losses
    return model

def eval_model(model, X, Y, metric,config):
    model.to(config.device)
    model.eval()
    X = X.to(config.device)
    Y = Y.to(config.device)
    
    with torch.no_grad():
        Y_pred = model(X)
        loss = metric(Y_pred,Y)
        #model.fix_parameters()
        return loss

    

class F1_Loss:
    def __init__(self,average='weighted'):
        self.average = average
        
    def __call__(self,Y_pred,Y):
        Y_pred = torch.argmax(Y_pred,dim=1)
        return f1_score(Y,Y_pred,average=self.average)

class Accuracy_Loss:
    def __call__(self,Y_pred,Y):
        Y_pred = torch.argmax(Y_pred,dim=1)
        return (Y_pred == Y).float().mean()
    

class R2_Loss:
    def __call__(self,Y_pred,Y):
        return r2_score(Y,Y_pred)
    