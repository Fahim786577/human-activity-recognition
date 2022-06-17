import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def makeDataframe(data):
    df = pd.DataFrame(data=data)
    df.plot(x="Activities", y=[ "After_Augmentation","Before_Augmentation"], kind="bar")
    sns.set(style="white", context="talk")
    plt.show()

UCIActivities = ["WALKING","UPSTAIRS","DOWNSTAIRS","SITTING","STANDING","LAYING","STAND_TO_SIT","SIT_TO_STAND","SIT_TO_LIE","LIE_TO_SIT","STAND_TO_LIE","LIE_TO_STAND"]
UCILabels = [label.capitalize() for label in UCIActivities]
RWLabels = ["Walking","Upstairs","Downstairs","Sitting","Standing","Laying"]
WISDMLabels = ["Jogging","Walking","UpStairs","DownStairs","Sitting","Standing"]


UCIdata ={
    'Activities':UCILabels,
    'Before_Augmentation':[15.49,14.26,13.04,15.41,17.16,16.84,1.14,1.05,1.35,1.27,1.76,1.18],
    'After_Augmentation':[10.78,9.92,9.07,10.72,11.94,11.72,5.96,5.96,5.96,5.96,5.96,5.96]  
}

WISDMdata ={
    'Activities':WISDMLabels,
    'Before_Augmentation':[30.04,37.92,11.57,9.81,6.45,4.21],
    'After_Augmentation':[20.89,26.37,13.19,13.19,13.19,13.19]  
}

RWdata ={
    'Activities':RWLabels,
    'Before_Augmentation':[21.88,8.98,7.45,20.53,20.74,20.43],
    'After_Augmentation':[20.75,10.37,10.37,19.47,19.67,19.37]  
}
acts = np.arange(12)
UCIdf = makeDataframe(UCIdata)
WISDMdf = makeDataframe(WISDMdata)
RWdf = makeDataframe(RWdata)
#print(acts)
#print(WISDMdf)
#print(RWdf)
"""df = pd.DataFrame({
    'Activities':RWLabels,
    'Before_Augmentation':[21.88,8.98,7.45,20.53,20.74,20.43],
    'After_Augmentation':[20.75,10.37,10.37,19.47,19.67,19.37]  
})
df.plot(x="Activities", y=["Before_Augmentation", "After_Augmentation"], kind="bar")"""
#plt.bar(acts -0.2,UCIdata['Before_Augmentation'],0.5)
#plt.bar(acts +0.2,UCIdata['After_Augmentation'],0.5)
#plt.show()