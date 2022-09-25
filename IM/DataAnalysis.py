import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd


#df= pd.read_csv("data/brain_tumor_dataset_features.csv");f=["NormalFeatures","AbNormalFeatures"]
#sns.displot(df[f], kind="kde", fill=True)
#sns.histplot(data=df[f], kde=True)

fig, axs = plt.subplots(ncols=2)
f=['Region1','Region2','Region3','Region4']
for i,path in enumerate(["BrainDataset-Normal","BrainDataset-AbNormal"]):
    df= pd.read_csv("data/"+path+".csv")
    sns.histplot(data=df['Region4'], kde=True, ax=axs[i])
    #sns.displot(df['Region1'], kind="kde", fill=True)

plt.grid(True)
plt.show()