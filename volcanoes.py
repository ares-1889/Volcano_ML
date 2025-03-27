import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('The_Volcanoes_Of_Earth.csv')

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

def removing_s(value):
   if isinstance(value,str):
      return value.split('(')[0]

def unique_values_count(df, column_name):
   return df[column_name].nunique()

def unique_values(df, column_name):
   return df[column_name].unique()

le = LabelEncoder()   

# Die Entfernung unnötigen Spalten
df = df.drop(['Volcano_Name','Volcano_Image','Latitude','Longitude','population_within_5km','population_within_100km','population_within_10km','population_within_30km','Country','Region','Subregion','epoch_period'],axis=1)

# Cleaning von Data , bzw : Entfernung von Einheit "m" und Daten auf "Feet"
df['Summit_and_Elevatiuon'] = df['Summit_and_Elevatiuon'].apply(removeFtandM)

# Datum auf die Integer-zahllinie abbgebildet 
df['Last_Eruption'] = df['Last_Eruption'].apply(date_to_int)

df['Volcano_Type'] = df['Volcano_Type'].apply(removing_s)

le.fit(df['Volcano_Type'])
le.classes_
df['Volcano_Type'] = le.transform(df['Volcano_Type'])


# print(df.head())
unique_count = unique_values_count(df,'Volcano_Type')
unique_vals = unique_values(df,'Volcano_Type')
# print(unique_vals)

df['Last_Eruption'] = pd.to_numeric(df['Last_Eruption'],errors='coerce')
df = df.dropna(subset=['Last_Eruption'])

# print(df.describe(include='all'))
correlation = df.corr()

plt.figure(figsize=(10, 5))
sns.histplot(df['Summit_and_Elevatiuon'], kde=True)
plt.title('Verteilung der Summit_and_Elevatiuon')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Summit_and_Elevatiuon'])
plt.title('Boxplot der Summit_and_Elevatiuon')
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Korrelation Heatmap')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x='Last_Eruption', y='Summit_and_Elevatiuon', data=df, hue='Volcano_Type')
plt.title('Streudiagramm von Last_Eruption und Summit_and_Elevatiuon')
plt.show()

sns.pairplot(df)
plt.show()

features_vectors = df[['Volcano_Type','Summit_and_Elevatiuon']]
target_vector = df['Last_Eruption']

X_train, X_test, y_train, y_test = train_test_split(features_vectors,target_vector,test_size = 0.2, random_state = 42)
rf  = RandomForestClassifier(n_estimators =100,random_state = 42)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
#0.01
