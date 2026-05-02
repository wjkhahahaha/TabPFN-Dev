import sys
import numpy as np
import  pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabpfn import TabPFNRegressor
import math
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

def  mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


import warnings
warnings.filterwarnings("ignore")

import random
import torch
import numpy as np

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)



class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


######################################################
#########################Chla#########################
######################################################

sys.stdout = DualOutput("./Chla_output_log.txt")


df_data = pd.read_csv('data.csv',sep=',',header=0, index_col=0)
X = df_data.iloc[:,:-5]
y = df_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=42)


reg = TabPFNRegressor(ignore_pretraining_limits=True)
reg.fit(X_train, y_train)

TabPFN_y_train_pred = reg.predict(X_train)
print('\nTrain mse:',mean_squared_error(TabPFN_y_train_pred,y_train))
print('Train rmse:',np.sqrt(mean_squared_error(TabPFN_y_train_pred,y_train)))
print('Train mae:',mean_absolute_error(TabPFN_y_train_pred,y_train))
print('Train mape:',mape(TabPFN_y_train_pred,y_train))
print('Train R2:',r2_score(y_train,TabPFN_y_train_pred))

TabPFN_y_test_pred = reg.predict(X_test)
print('\nTest mse:',mean_squared_error(TabPFN_y_test_pred,y_test))
print('Test rmse:',np.sqrt(mean_squared_error(TabPFN_y_test_pred,y_test)))
print('Test mae:',mean_absolute_error(TabPFN_y_test_pred,y_test))
print('Test mape:',mape(TabPFN_y_test_pred,y_test))
print('Test R2:',r2_score(y_test,TabPFN_y_test_pred))

x_explain_multiple = X_test.values 
n_samples = len(X_test) 
feature_names = ['NH3-N','TP','TN','COD','pH','WT','DO','EC','Tur']
all_feature_importance = np.zeros((n_samples, len(feature_names)))


print("\nCalculating SHAP values for feature importance...")
# Get a TabPFNExplainer
explainer = tabpfn_shapiq.get_tabpfn_explainer(model=reg,data=X,labels=y,index="SV", verbose=True)


for i in range(n_samples):
    print(f"Processing sample {i+1}/{n_samples}...")
    shapley_values = explainer.explain(x=x_explain_multiple[i], budget=100)
    all_feature_importance[i] = np.abs(shapley_values.values[1:])


feature_importance = np.mean(all_feature_importance, axis=0)
print(feature_importance)


plt.figure(figsize=(5, 5))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.xlabel('Mean |SHAP value|')
plt.title('Feature Importance Based on SHAP Values')
plt.tight_layout()
plt.savefig('Chla_feature_importance.svg', bbox_inches='tight')

sys.stdout.log.close() 
sys.stdout = sys.stdout.terminal 
