import pandas as pd
import numpy as np
import pickle

import imodels
from torchvision.datasets import MNIST
from PIL import Image
from scipy.io import arff
import warnings
from pmlb import fetch_data
warnings.filterwarnings("ignore")

import sys
one_hot = True
datapath = "data"

def column_to_numeric(series):
    values = series.unique()
    mapping = {}
    for ind,v in enumerate(values):
        mapping[v] = ind
    return mapping
    

# German Credit Dataset (Binary Classification Risk/No Risk)
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
def load_credit():
    df = pd.read_csv(datapath+"/credit/german_credit_data.csv",sep=",",index_col=0)
    df.dropna(inplace=True)
    
    df["Risk"].replace( {"good":"Low Risk","bad":"High Risk"},inplace=True)
    df["Job"].replace({0 : "unskilled non-resident" , 1 : "unskilled resident", 2 : "skilled", 3 : "highly skilled"},inplace=True)
    
    df["Risk"].replace( {"Low Risk":0,"High Risk":1},inplace=True)
    df = pd.get_dummies(df,columns=["Job","Housing","Saving accounts","Checking account","Purpose"],dtype=int)
    
    df['Age'] = df['Age'].astype('int')
    df['Duration'] = df['Duration'].astype('int')
    df['Credit amount'] = df['Credit amount'].astype('int')
    
    output = {}
    output["df"] = df.copy(True)
    output["target"] = df["Risk"].to_numpy()
    df = df.drop("Risk",axis=1) 
    output["mapper"] = {}
    output["target_name"] = "Risk of Default"
    turn_to_numeric = list(filter(lambda x: not pd.api.types.is_numeric_dtype(df[x]),df.columns.values.tolist()))
    for col in turn_to_numeric:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)   
        index = df.columns.values.tolist().index(col)
        output["mapper"][index] = {v: k for k, v in replacement.items()}
    
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output

def load_default():
    df = pd.read_csv(datapath+"/default/default.csv",sep=",",index_col=0)
    df.dropna(inplace=True)
    for i in range(1,7):
        df.drop("PAY_{:d}".format(i),axis=1,inplace=True)
        df.drop("PAY_AMT{:d}".format(i),axis=1,inplace=True)
        df.drop("BILL_AMT{:d}".format(i),axis=1,inplace=True)
    output = {}
    df["CREDIT LIMIT"] = (df["CREDIT LIMIT"]/33).round()
    df["Default"].replace( {0:"No Default",1:"Default"},inplace=True)
    df["SEX"].replace({1:"male",2:"female"},inplace=True)
    df["EDUCATION"].replace({1:"graduate school",2:"university",3:"high school",4:"other"},inplace=True)
    df["MARRIAGE"].replace({1:"married",2:"single",3:"other"},inplace=True)
    output["df"] = df.copy(True)
    df["Default"].replace( {"No Default":0,"Default":1},inplace=True)
    output["target"] = df["Default"].to_numpy()
    df = df.drop("Default",axis=1) 
    output["mapper"] = {}
    output["target_name"] = "Risk of Default"
    turn_to_numeric = list(filter(lambda x: not pd.api.types.is_numeric_dtype(df[x]),df.columns.values.tolist()))
    for col in turn_to_numeric:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)   
        index = df.columns.values.tolist().index(col)
        output["mapper"][index] = {v: k for k, v in replacement.items()}
    
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output

def load_wages():
    df = pd.read_csv(datapath+"/wages/wages.csv")
    df.dropna(inplace=True)
    df = pd.get_dummies(df,columns=["race"],dtype=int)
    output = {}
    output["df"] = df.copy(True)
    for col in ["sex"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)
    output["target"] = df["earn"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("earn")
    output["target_name"] = "income"
    output["data"] = df.drop("earn",axis=1).to_numpy()

    return output


def load_student():
    df = pd.read_csv(datapath+"/student/student-por.csv",sep=";")
    df.dropna(inplace=True)
    df = pd.get_dummies(df,columns=["Pstatus","Medu","Fedu","Mjob","Fjob","reason"],dtype=int)
    output = {}
    df = df.drop(["G1","G2"],axis=1)
    output["df"] = df.copy(True)
    output["target"] = df["G3"].to_numpy()
    df = df.drop("G3",axis=1)
    output["target_name"] = "Grade"
    turn_to_numeric = list(filter(lambda x: not pd.api.types.is_numeric_dtype(df[x]),df.columns.values.tolist()))
    for col in turn_to_numeric:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)    
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output

def load_bike():
    df = pd.read_csv(datapath+"/bike/day.csv",index_col=0)
    df = pd.get_dummies(df,columns=["season"],dtype=int)
    df = df.drop(["instant","dteday","casual","registered","yr"],axis=1)

    df.dropna(inplace=True)
    output = {}
    output["df"] = df.copy(True)

    output["target"] = df["cnt"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("cnt")
    output["data"] = df.drop("cnt",axis=1).to_numpy()
    output["target_name"] = "Bike rentals"
    return output


def load_life():
    df = pd.read_csv(datapath+"/life-expectancy/life.csv")
    df.dropna(inplace=True)
    output = {}
    df.drop(["Country"],axis=1,inplace=True)
    output["df"] = df.copy(True)
    for col in ["Status"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)
    output["target"] = df["Life expectancy"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("Life expectancy")
    output["data"] = df.drop("Life expectancy",axis=1).to_numpy()
    output["target_name"] = "Life expectancy"
    return output

def load_adult():
    df = pd.read_csv(datapath+"/adult/adult.data")
    output = {}
    output["df"] = df.copy(True)
    df["class"].replace([" <=50K"," >50K"],[0,1],inplace=True)
    for col in ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)

    output["target"] = df["class"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("class")
    output["data"] = df.drop("class",axis=1).to_numpy()
    output["target_name"] = "Income >= 50k"

    return output

def load_insurance():
    df = pd.read_csv(datapath+"/insurance/insurance.csv")
    output = {}
    df['age'] = df['age'].astype('int')
    df['charges'] = df['charges'].astype('int')
    df['bmi'] = df['bmi'].astype('int')
    output["df"] = df.copy(True)
    df["smoker"].replace({"yes":1,"no":0},inplace=True)
    df["sex"].replace({"male":1,"female":0},inplace=True)
    df = pd.get_dummies(df,columns=["region"],dtype=int)

    output["target"] = df["charges"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("charges")
    output["data"] = df.drop("charges",axis=1).to_numpy()
    output["target_name"] = "Insurance Rate"
    return output
    
# OS overall survival, DFI disease free interval, PFI progression free interval, DSS disease specific survival
def load_cancer():
    cancer = pd.read_csv(datapath+'/cancer/tcga_clinical_data.csv')
    features = ["OS_days","Mutation.Count","Fraction.Genome.Altered","Aneuploidy.Score","Buffa.Hypoxia.Score","Ragnum.Hypoxia.Score","Winter.Hypoxia.Score","Keck_Classification","HPV","HPV16","Alcohol","Smoking","Gender","Subsite","Age","Grading","clinical_Stage3.4","cT","cN","cM","Neoadjuvant_Treatment","Radiation","Smoking_status","Anatomic_subdivision","Vital_status","Race","clinical_stage","pT","pN","pathologic_stage","postoperative_rx_tx","margin_status","Extracapsular_spread","presence_of_pathological_nodal_extracapsular_spread","prior_dx","History_of_Malignancy","hpv_status_by_p16_testing","hpv_status_by_ish_testing","primary_therapy_outcome_success","lymph_node_examined_count","primary_lymph_node_presentation_assessment","stopped_smoking_year","year_of_tobacco_smoking_onset","acronym","icd_10","laterality","additional_pharmaceutical_therapy","additional_radiation_therapy","tissue_retrospective_collection_indicator","icd_o_3_histology","new_neoplasm_event_occurrence_anatomic_site","lymphnode_dissection_method_right","lymphnode_dissection_method_left","tissue_prospective_collection_indicator","icd_o_3_site","lymphnode_neck_dissection","new_tumor_event_after_initial_treatment","ethnicity","person_neoplasm_cancer_status","year_of_initial_pathologic_diagnosis","histological_type","tissue_source_site","number_of_lymphnodes_positive"]
    features = ["OS_days","Mutation.Count","Fraction.Genome.Altered","Aneuploidy.Score","Buffa.Hypoxia.Score","Ragnum.Hypoxia.Score","Winter.Hypoxia.Score","Keck_Classification","HPV","HPV16","Alcohol","Smoking","Gender","Subsite","Age","Grading","clinical_Stage3.4","cT","cN","cM","Neoadjuvant_Treatment","Radiation","Smoking_status","Anatomic_subdivision","Vital_status","Race","clinical_stage"]
    cancer = cancer[features]
    output = {}
    output["df"] = cancer.copy(True)
    # count how many null values are in each column
    cancer = cancer.dropna()
    output["target"] = cancer["OS_days"]
    cancer.drop("OS_days",axis=1,inplace=True)
    output["target_name"] = "Overall Survival (days)"
    # go through columns with non-numeric values and replace them with numeric values
    for col in cancer.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(cancer[col]):
            replacement = column_to_numeric(cancer[col])
            cancer[col].replace(replacement,inplace=True)
    output["data"] = cancer.to_numpy()
    output["feature_names"] = cancer.columns.values.tolist()

    return output

def load_smoking():
    smoking = pd.read_csv(datapath+'/smoking/smoking.csv')
    smoking.dropna(inplace=True)
    smoking.drop("ID",axis=1,inplace=True)
    features = smoking.columns.values.tolist()
    target_name = ["relaxation","Cholesterol"]
    for f in target_name:
        features.remove(f)
    output = {}
    output["df"] = smoking.copy(True)
    output["target"] = smoking[target_name].to_numpy()
    smoking.drop(target_name,axis=1,inplace=True)
    output["target_name"] = target_name
    # go through columns with non-numeric values and replace them with numeric values
    for col in smoking.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(smoking[col]):
            replacement = column_to_numeric(smoking[col])
            smoking[col].replace(replacement,inplace=True)
    output["data"] = smoking.to_numpy()
    output["feature_names"] = smoking.columns.values.tolist()
    return output

# def load_diabetes():
#     diabetes = pd.read_csv(datapath+'/diabetes/diabetes_prediction_dataset.csv')
#     features = diabetes.columns.values.tolist()
#     target_name = ["blood_glucose_level","HbA1c_level"]
#     for f in target_name:
#         features.remove(f)
#     output = {}
#     output["df"] = diabetes.copy(True)
#     output["target"] = diabetes[target_name].to_numpy()
#     diabetes.drop(target_name,axis=1,inplace=True)
#     for col in diabetes.columns.values.tolist():
#         if not pd.api.types.is_numeric_dtype(diabetes[col]):
#             replacement = column_to_numeric(diabetes[col])
#             diabetes[col].replace(replacement,inplace=True)
#     output["data"] = diabetes.to_numpy()
#     output["feature_names"] = diabetes.columns.values.tolist()
#     return output

def load_wine():
    wine = pd.read_csv(datapath+'/wine-quality/winequality-white.csv',sep=";")
    wine.dropna(inplace=True)
    features = wine.columns.values.tolist()
    target_name = ["quality"]
    for f in target_name:
        features.remove(f)
    output = {}
    output["df"] = wine.copy(True)
    output["target"] = wine[target_name].to_numpy()
    wine.drop(target_name,axis=1,inplace=True)
    for col in wine.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(wine[col]):
            replacement = column_to_numeric(wine[col])
            wine[col].replace(replacement,inplace=True)
    output["data"] = wine.to_numpy()
    output["feature_names"] = wine.columns.values.tolist()
    return output

def load_covid():
    df = pd.read_csv(datapath+"/covid/covid.csv")
    output = {}
    df.drop(["outcome","id","patient_id","weekday_change_of_status","hour_change_of_status","weekday_admit","hour_admit","days_change_of_status","date_admit","date_change_of_status","hospital"],axis=1,inplace=True)
    df.drop(df[df["group"]=="Patient"].index,inplace=True)
    df.dropna(inplace=True)
    output["df"] = df.copy(True)
    df["group"].replace(["Expired","Discharged"],[0,1],inplace=True)
    for col in ["sex","race"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)
    df.dropna(inplace=True)
    output["target"] = df["group"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("group")
    df.drop("group",axis=1,inplace=True)
    output["data"] = df.to_numpy()
    output["target_name"] = "group"
    return output

def load_forest():
    forest = pd.read_csv(datapath+'/forest/forestfires.csv',sep=",")
    forest.dropna(inplace=True)
    forest = pd.get_dummies(forest,columns=["month","day"],dtype=int)
    features = forest.columns.values.tolist()
    target_name = ["area"]
    for f in target_name:
        features.remove(f)
    output = {}
    output["df"] = forest.copy(True)
    output["target"] = forest[target_name].to_numpy()
    forest.drop(target_name,axis=1,inplace=True)
    for col in forest.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(forest[col]):
            replacement = column_to_numeric(forest[col])
            forest[col].replace(replacement,inplace=True)
    output["data"] = forest.to_numpy()
    output["feature_names"] = forest.columns.values.tolist()
    return output



def load_diabetes_classification():
    diabetes = pd.read_csv(datapath+'/diabetes/diabetes_prediction_dataset.csv')
    features = diabetes.columns.values.tolist()
    diabetes.dropna(inplace=True)
    target_name = ["diabetes"]
    for f in target_name:
        features.remove(f)
    output = {}
    output["df"] = diabetes.copy(True)
    output["target"] = diabetes[target_name].to_numpy()
    diabetes.drop(target_name,axis=1,inplace=True)
    for col in diabetes.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(diabetes[col]):
            replacement = column_to_numeric(diabetes[col])
            diabetes[col].replace(replacement,inplace=True)
    output["data"] = diabetes.to_numpy()
    output["feature_names"] = diabetes.columns.values.tolist()
    return output

def load_boston():
    boston = pd.read_csv(datapath+'/boston_housing/HousingData.csv')
    features = boston.columns.values.tolist()
    boston.dropna(inplace=True)
    target_name = ["MEDV"]
    for f in target_name:
        features.remove(f)
    output = {}
    output["df"] = boston.copy(True)
    output["target"] = boston[target_name].to_numpy()
    boston.drop(target_name,axis=1,inplace=True)
    for col in boston.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(boston[col]):
            replacement = column_to_numeric(boston[col])
            boston[col].replace(replacement,inplace=True)
    output["data"] = boston.to_numpy()
    output["feature_names"] = boston.columns.values.tolist()
    return output

def load_cpu():
    cpu = pd.read_csv(datapath+'/cpu/machine.data')
    cpu.dropna(inplace=True)
    cpu.columns = ["vendor","model","MYCT","MMIN","MMAX","CACH","CHMIN","CHMAX","PRP","ERP"]
    cpu.drop(["model"],axis=1,inplace=True)
    cpu = pd.get_dummies(cpu,columns=["vendor"],dtype=int)
    features = cpu.columns.values.tolist()
    target_name = ["ERP"]
    for f in target_name:
        features.remove(f)
    output = {}
    output["df"] = cpu.copy(True)
    output["target"] = cpu[target_name].to_numpy()
    cpu.drop(target_name,axis=1,inplace=True)
    for col in cpu.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(cpu[col]):
            replacement = column_to_numeric(cpu[col])
            cpu[col].replace(replacement,inplace=True)
    output["data"] = cpu.to_numpy()
    output["feature_names"] = cpu.columns.values.tolist()
    return output

def load_abalone():
    df = pd.read_csv(datapath+"/abalone/abalone.data",header=None)
    df.dropna(inplace=True)
    df.columns = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","age"]
    df = pd.get_dummies(df,columns=["sex"],dtype=int)
    output = {}
    target_name = ["age"]
    output["df"] = df.copy(True)
    output["target"] = df[target_name].to_numpy()
    df.drop(target_name,axis=1,inplace=True)
    
    for col in df.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(df[col]):
            replacement = column_to_numeric(df[col])
            df[col].replace(replacement,inplace=True)
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output
    
def load_automobiles():
    auto = pd.read_csv(datapath+'/automobile/automobile.data',sep=",",header=None)
    
    auto.columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location",
    "wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio",
    "horsepower","peak-rpm","city-mpg","highway-mpg","price"]
    auto.drop(["normalized-losses"],axis=1,inplace=True)
    auto.dropna(inplace=True)
    df = pd.get_dummies(auto,columns=["make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","engine-type","fuel-system"],dtype=int)
    output = {}
    target_name = ["price"]
    output["df"] = df.copy(True)
    output["target"] = df[target_name].to_numpy()
    df.drop(target_name,axis=1,inplace=True)

    for col in df.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(df[col]):
            replacement = column_to_numeric(df[col])
            df[col].replace(replacement,inplace=True)
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output

def load_air_quality():
    air = pd.read_csv(datapath+'/airquality/AirQualityUCI.csv',sep=";")
    air = air.drop(columns=['Unnamed: 15','Unnamed: 16'])
    air.dropna(inplace=True)
    # process data into month, day and hour integers
    air["Date"] = pd.to_datetime(air["Date"], dayfirst=True)
    air["month"] = air["Date"].dt.month
    air["day"] = air["Date"].dt.day
    air["hour"] = air["Time"].str.split(".",expand=True)[0].astype(int)
    air.drop(["Date","Time"],axis=1,inplace=True)
    output = {}
    target_name = "CO(GT)"
    air[target_name] = air[target_name].replace(",",".",regex=True).astype('float')
    output["df"] = air.copy(True)
    target = air[target_name].to_numpy()
    mask = target>=0
    output["target"] = air[target_name].to_numpy()[mask]
    air.drop(target_name,axis=1,inplace=True)
    for col in air.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(air[col]):
            replacement = column_to_numeric(air[col])
            air[col].replace(replacement,inplace=True)
    output["data"] = air.to_numpy()[mask]
    output["feature_names"] = air.columns.values.tolist()
    return output

def load_liver():
    liver = pd.read_csv(datapath+'/liver/bupa.data',sep=",")
    liver.columns = ["mcv","alkphos","sgpt","sgot","gammagt","drinks","selector"]
    liver.drop(["selector"],axis=1,inplace=True)
    liver.dropna(inplace=True)
    output = {}
    target_name = "drinks"
    output["df"] = liver.copy(True)
    output["target"] = liver[target_name].to_numpy()
    liver.drop(target_name,axis=1,inplace=True)
    for col in liver.columns.values.tolist():
        if not pd.api.types.is_numeric_dtype(liver[col]):
            replacement = column_to_numeric(liver[col])
            liver[col].replace(replacement,inplace=True)
    output["data"] = liver.to_numpy()
    output["feature_names"] = liver.columns.values.tolist()
    return output

def load_gold(forsg=False):
    gold = pd.read_csv("../data/gold/data_Au5-14_2.0.0.csv")
    cols = pd.read_csv("../data/gold/data_Au5-14_2.0.0_attributes.csv")
    gold.columns = ["Au+N",*list(cols["Structure"])]

    Y = gold["HOMO-LUMO"].to_numpy().reshape(-1,1)
    gold["N-even"] = (gold["N"] % 2) == 0
    gold = gold.drop(["Au+N","-LUMO","-HOMO","HOMO-LUMO","chemical hardness",
                    "electronic chemical potential","Evdw/N","Evdw-Evdw0","hardness-hardness0","electronic chemical pot. - electronic chemical pot0","|F| / N"],axis=1)
    cols = gold.columns
    #print(gold.dtypes)
    if forsg:
        return  {"data":gold, "target":Y, "feature_names":gold.columns}
    else:
        X = gold.to_numpy()
        return  {"data":X, "target":Y, "feature_names":gold.columns}

def load_goldvdw(forsg=False):
    gold = pd.read_csv("../data/gold/data_Au5-14_2.0.0.csv")
    cols = pd.read_csv("../data/gold/data_Au5-14_2.0.0_attributes.csv")
    gold.columns = ["Au+N",*list(cols["Structure"])]
    Y = gold["Evdw-Evdw0"].to_numpy().reshape(-1,1)
    gold["N-even"] = (gold["N"] % 2) == 0
    gold = gold.drop(["Au+N","-LUMO","-HOMO","HOMO-LUMO","chemical hardness",
                  "electronic chemical potential","Evdw/N","Evdw-Evdw0","hardness-hardness0","electronic chemical pot. - electronic chemical pot0","|F| / N"],axis=1)
    cols = gold.columns
    #print(gold.dtypes)
    if forsg:
        return  {"data":gold, "target":Y, "feature_names":gold.columns}
    else:
        X = gold.to_numpy()
        return  {"data":X, "target":Y, "feature_names":gold.columns}

def to_numpy(data, vars=['lep_pt','lep_E','lep_phi','lep_eta',"mllll"]): #''lep_charge','lep_type'
    import awkward as ak
    data = data[ak.count(data.lep_type,axis=1)==4]
    final = []
    cols = []
    for var in vars:
        col = getattr(data, var)
        col_numpy = ak.to_numpy(col)
        if len(col_numpy.shape) == 1:
            col_numpy = np.expand_dims(col_numpy,axis=1)
            cols.append(var)
        else:
            cols.extend([var + "_"+str(j+1) for j in range(col_numpy.shape[1])])
        final.append(col_numpy)
    final = np.concatenate(final, axis=1)

    return final,cols

def get_table(data,keys=[ ('Background $ZZ^*$',"z"), ('Signal ($m_H$ = 125 GeV)',"s")],s_portion=1.0): #('Background $Z,t\\bar{t}$',"b"),
    table = []
    labels = []
    for k,l in keys:
        final,cols = to_numpy(data[k])
        if l == "s":
            n_samples = int(len(final)*s_portion)
            final = final[0:n_samples]
        table.append(final)
        labels.extend(final.shape[0]*l)
    return np.concatenate(table,axis=0), cols, np.array(labels)

def load_higgs(s_portion=0.5, mask=False):
    try:
        import awkward as ak
    except:
        import os
        os.system('pip install vector uproot3 requests aiohttp awkward uproot')

    with open('../data/higgs/HZZ_MC.pkl', 'rb') as f:
        data = pickle.load(f) 
    test, cols, labels = get_table(data,s_portion=s_portion)
    X = test[:,0:-1]
    Y = test[:,-1]
    if mask:
        mask = Y>95
        Y = Y[mask].reshape(-1,1)
        X = X[mask]
        labels = labels[mask]
        return {"data":X, "target":Y, "feature_names":cols[0:-1], "labels":labels}
    else:
        X = test[:,0:-1]
        Y = test[:,-1].reshape(-1,1)
        return {"data":X, "target":Y, "feature_names":cols[0:-1], "labels":labels}

def load_imodels_data(name):
    X,Y, feature_names = imodels.get_clean_dataset(name)
    return {"data":X, "target":Y, "feature_names":feature_names}

def load_pmlb_data(name):
    ds = fetch_data(name, local_cache_dir=datapath)
    Y = ds['target'].values
    Y = (Y==Y.max()).astype(int)
    X = ds.drop(columns=['target']).values
    feature_names = ds.drop(columns=['target']).columns
    return {"data":X, "target":Y, "feature_names":feature_names}

def load_multi_class(name):
    ds = fetch_data(name) #yeast
    Y = ds['target'].values
    Y = (Y - Y.min()).astype(int)
    
    #print(name)
    y_re = np.zeros(Y.max()+1).astype(int)
    #print(np.unique(Y))
    for enu, i in enumerate(np.unique(Y)):
        y_re[i] = enu
    Y = y_re[Y]
    #print(np.unique(Y))
    #print(y_re)
    
    #sys.exit()
    X = ds.drop(columns=['target']).values
    feature_names = ds.drop(columns=['target']).columns
    return {"data":X, "target":Y, "feature_names":feature_names}

def load_pmlb_data_multi(name):
    ds = fetch_data(name, local_cache_dir=datapath)
    Y = ds['target'].values
    Y = (Y==Y.max()).astype(int)
    X = ds.drop(columns=['target']).values
    feature_names = ds.drop(columns=['target']).columns
    return {"data":X, "target":Y, "feature_names":feature_names}

def load_mnist():
    pytorch_dataset = MNIST(root=datapath,download=True)
    # turn it into a numpy array
    
    X = pytorch_dataset.data.numpy()
    
    # downscale to 14x14
    X = X.reshape(-1,28,28)
    #X = np.array([np.array(Image.fromarray(x).resize((14,14))) for x in X])
    #X = X.reshape(-1,14*14)

    labels = pytorch_dataset.targets.numpy()
    # filter out 0 and 1
    mask = np.ones(len(X),dtype=bool)
    mask = (labels == 0) | (labels == 1)
    
    X = X[mask]
    labels = labels[mask]
    # random shuffle
    mask = np.arange(len(X))
    np.random.shuffle(mask)
    X = X[mask][:1000,:]
    labels = labels[mask][:1000]
    feature_names = ["pixel_{:d}".format(i) for i in range(28*28)]
    return {"data":X, "target":labels, "feature_names":feature_names}

def load_ozone_level():
    arff_file = arff.loadarff(datapath+'/ozone-level/ozone-level.arff')

    df = pd.DataFrame(arff_file[0])
    labels = (df["Class"]==b'1').astype(int)
    df = df.drop("Class",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}

def load_madelon():
    arff_file = arff.loadarff(datapath+'/madelon/madelon.arff')

    df = pd.DataFrame(arff_file[0])
    labels = (df["Class"]==b'1').astype(int)
    df = df.drop("Class",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}

def load_pc1():
    arff_file = arff.loadarff(datapath+'/pc1/pc1.arff')

    df = pd.DataFrame(arff_file[0])
    labels = (df["defects"]==b'true').astype(int)
    df = df.drop("defects",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}

def load_phoneme():
    arff_file = arff.loadarff(datapath+'/phoneme/phoneme.arff')

    df = pd.DataFrame(arff_file[0])
    labels = (df["Class"]==b"2").astype(int)
    df = df.drop("Class",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}

def load_qsar_biodeg():
    arff_file = arff.loadarff(datapath+'/qsar-biodeg/qsar-biodeg.arff')

    df = pd.DataFrame(arff_file[0])
    labels = (df["Class"]==b"2").astype(int)
    df = df.drop("Class",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}

def load_eeg_eye_state():
    arff_file = arff.loadarff(datapath+'/eeg-eye-state/eeg-eye-state.arff')

    df = pd.DataFrame(arff_file[0])
    labels = (df["Class"]==b"2").astype(int)
    df = df.drop("Class",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}

def load_electricity():
    arff_file = arff.loadarff(datapath+'/electricity/electricity.arff')

    df = pd.DataFrame(arff_file[0])
    labels = (df["class"]==b"UP").astype(int)
    df = df.drop("class",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}

def load_phishing():

    arff_file = arff.loadarff(datapath+'/phishing/phishing.arff')

    df = pd.DataFrame(arff_file[0])
    labels = (df["Result"]==b"1").astype(int)
    df = df.drop("Result",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}

def load_android():

    df = pd.read_csv(datapath+'/android-permission/data.csv')
    
    labels = (df["Result"]).astype(int)
    df = df.drop("Result",axis=1)
    X = df.to_numpy()
    feature_names = df.columns.values.tolist()

    return {"data":X, "target":labels, "feature_names":feature_names}
 