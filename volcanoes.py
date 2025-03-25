import pandas as pd 
import matplotlib as mlt
import seaborn as sns

def removeFtandM(value):
    if isinstance(value,str):
      return value.split('m')[0]
    return float(value)

def date_to_int(value):
   if isinstance(value, str):
      if 'BCE' in value:
        return -int(value.replace('BCE','').strip())
      elif 'CE' in value:
         return int(value.replace('CE','').strip())
      return value
   
   
   
df = pd.read_csv('The_Volcanoes_Of_Earth.csv')

df = df.drop(['Volcano_Name','Volcano_Image','Latitude','Longitude','population_within_5km','population_within_100km','population_within_10km','population_within_30km'],axis=1)

df['Summit_and_Elevatiuon'] = df['Summit_and_Elevatiuon'].apply(removeFtandM)
df['Last_Eruption'] = df['Last_Eruption'].apply(date_to_int)

print(df.head())





