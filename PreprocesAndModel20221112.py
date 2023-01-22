######################################################################## PACKAGES #####################################################################

from operator import concat
from pydoc import describe
from statistics import mean
from unicodedata import numeric
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn.inspection import permutation_importance

######################################################################## PACKAGES #####################################################################

######################################################################## PREPROCESSING ################################################################

data_source1 = pd.read_csv("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/offersLento09052022.csv")
data_source2 = pd.read_csv("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/offersLento28062022.csv")
data_source3 = pd.read_csv("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/offersLento18082022.csv")
data_source4 = pd.read_csv("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/offersLento16092022.csv")
data_source5 = pd.read_csv("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/offersLento11102022.csv")
data_source6 = pd.read_csv("C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/offersLento12112022.csv")

data = pd.concat([data_source1, data_source2], ignore_index=True)
data = pd.concat([data, data_source3], ignore_index=True)
data = pd.concat([data, data_source4], ignore_index=True)
data = pd.concat([data, data_source5], ignore_index=True)
data = pd.concat([data, data_source6], ignore_index=True)

data=data.drop_duplicates(subset=['offer_URL'],keep="first").reset_index(drop=True)
data=data.drop_duplicates(subset=['offer_header'],keep="first").reset_index(drop=True)

data = data.reset_index(drop=True)

car_cat=list(data['car_category'])
data['car_brand']=''
data['car_model']=''

z=0
for i in car_cat:
    s=str(i)
    s=s.split('/')
    s=str(s).replace(" ", "")
    s=s.replace('[', "")
    s=s.replace(']', "")
    s=s.replace('\'', "")
    s=s.split(',')
    data['car_brand'][z]=s[0]
    data['car_model'][z]=s[1]
    z=z+1

data['car_brand']
car_mil=list(data['car_mileage'])
car_cap=list(data['car_eng_capacity'])
car_price=list(data['car_price'])

data['price']=''


for i in range(0,len(car_price)):
    car_pricex=str(car_price[i]).replace(" ", "")
    car_pricex=str(car_pricex).replace("zł", "")
    car_pricex=str(car_pricex).replace(",", ".")
    print(i)
    if car_pricex.find("-") != -1:
        ind = car_pricex.index("-")
        data['price'][i]=car_pricex[0:ind]
    else:
         data['price'][i]=car_pricex

for i in range(0,len(data['price'])):
    data['price'][i]=str(data['price'][i]).replace(" ", "")
    data['price'][i]=str(data['price'][i]).replace(",", ".")
    data['price'][i]=re.sub('[A-Za-z]*', 'x', data['price'][i])
    data['price'][i]=str(data['price'][i]).replace("x", "")
    if data['price'][i] =='':
        data['price'][i]=float(0)
    else:
        data['price'][i]=float(data['price'][i])

data['price'] = pd.to_numeric(data['price']).astype(int)
data['price'].dtype

## delete wrong format price rows
data.iloc[27922,:]
data.iloc[38025,:]

data = data.drop(27922)
data = data.drop(38025)
data = data.reset_index(drop=True)


data['price'] = pd.to_numeric(data['price']).astype(int)
data['price'].dtype

## continue data transformation

for i in range(0,len(data['car_mileage'])):
    car_mil=str(data['car_mileage'][i]).replace(" ", "")
    car_mil=str(car_mil).replace("km", "")
    data['car_mileage'][i]=float(car_mil)

for i in range(0,len(data['car_eng_capacity'])):
    car_mil=str(data['car_eng_capacity'][i]).replace(" ", "")
    car_mil=str(car_mil).replace("cm3", "")
    data['car_eng_capacity'][i]=float(car_mil)

data['car_mileage'] = pd.to_numeric(data['car_mileage']).astype(int)
data['car_eng_capacity'] = pd.to_numeric(data['car_eng_capacity']).astype(int)
data['car_production_year'] = pd.to_numeric(data['car_production_year']).astype(int)


data['day_of_week_added']=''
data['day_added']=''
data['month_added']=''
data['year_added']=''
data['hour_added']=''
data['minute_added']=''
for i in range(len(data['date_added'])):
    split1 = str(data['date_added'][i]).split(',')
    split12 = str(split1[0]).split(':')
    split2 = str(split1[1]).split(' ')
    split3 = str(split1[2]).split(' ')
    split4 = str(split3[2]).split(':')
    data['day_of_week_added'][i] = split12[1].replace(" ", "")
    data['day_added'][i] = split2[1].replace(" ", "")
    data['month_added'][i] = split2[2].replace(" ", "")
    data['year_added'][i] = split3[1].replace(" ", "")
    data['hour_added'][i] = split4[0].replace(" ", "")
    data['minute_added'][i] = split4[1].replace(" ", "")

## Backup data

data_raw = data

# Remove not needed columns

data.columns

del data['promo_reg']
del data['promo_loc']
del data['Unnamed: 0']
del data['date_added']
del data['seller_URL']
del data['offer_URL']
del data['car_price']
del data['car_category']

del data['offer_header']
del data['offer_vieved']
del data['day_of_week_added']
del data['day_added']
del data['month_added']
del data['year_added']
del data['hour_added']
del data['minute_added']

data.columns

data_backup = data.copy()

######################################################################## PREPROCESSING ################################################################

######################################################################## TRAINING DATASET #############################################################

## split data to test and train

data_train, data_test = train_test_split(data, test_size = 0.25, random_state = 0)
data_train
data_test

## perform EDA only on train sample to make sure there is no data leakage

data = data_train.copy()

## Check types, shape etc.

data.shape
data.describe()
data.dtypes

## Check correlation - it seems that it is not correlated at all. Why? Probably there are outliers

data.corr()

## Check the data distribution - maybe there are some outliers

ax1 = data.plot.scatter(x='car_production_year',y='price',c='DarkBlue')
plt.show()

# It seems that price have a few big outliers - let us delete them. Let us also delete cars that costs more than 600k, as it seems to be natural barier

std_flag_price = data['price']<=600000
data['std_flag_price']=std_flag_price
data = data[data['std_flag_price']==True]
del data['std_flag_price']

std_flag_price = data['price']>=10000
data['std_flag_price']=std_flag_price
data = data[data['std_flag_price']==True]
del data['std_flag_price']

# Let us check the plot now. Now it is much better

ax1 = data.plot.scatter(x='car_production_year',y='price',c='blue')
plt.show()


# Now let us take a look at car production year. We may leave only cars were produced between 2000 and now

data.car_production_year.value_counts()
data['car_production_year'].hist(bins=30)
plt.show()

std_flag_year = data['car_production_year']>=2000
data['std_flag_year']=std_flag_year
data = data[data['std_flag_year']==True]
del data['std_flag_year']

ax1 = data.plot.scatter(x='car_production_year',y='price',c='DarkBlue')
plt.show()

# We can see that the pattern is more like polynomial order 2 not linear

# Now let us take a look at car mileage. There are some outliers. Let us then delete them as we did in car price

ax2 = data.plot.scatter(x='car_mileage',y='price',c='DarkBlue')
plt.show()

# Let us get rid of cars that have mileage over 500k - it is very unusual and we don`t want to look at such cars. Now it looks good.

std_flag_mil = (data['car_mileage'])<500000
data['std_flag_mil']=std_flag_mil
data = data[data['std_flag_mil']==True]
del data['std_flag_mil']

ax2 = data.plot.scatter(x='car_mileage',y='price',c='DarkBlue')
plt.show()

# Now let us look at end capacity. It seems that there are some nonsense data. Let us get rid of the cars that have eng cap below 1k cm^3 over 6k cm^3

ax3 = data.plot.scatter(x='car_eng_capacity',y='price',c='DarkBlue')
plt.show()

std_flag_cap = (data['car_eng_capacity'])>=1000
data['std_flag_cap']=std_flag_cap
data = data[data['std_flag_cap']==True]
del data['std_flag_cap']

std_flag_cap = (data['car_eng_capacity'])<=6000
data['std_flag_cap']=std_flag_cap
data = data[data['std_flag_cap']==True]
del data['std_flag_cap']

ax3 = data.plot.scatter(x='car_eng_capacity',y='price',c='DarkBlue')
plt.show()

data = data.reset_index(drop=True)

## Categorical data inspection

## car brand

sns.catplot(x='car_brand',
            data=data,
            kind='count')

plt.show()

freq_car_brand = data['car_brand'].value_counts()
freq_car_brand
data_len_treshold = int(len(data)/100)

len(freq_car_brand)
data['car_brand_country']=''

for i in range(0,len(data)):
    print(i)
    car_br = data['car_brand'][i]
    if car_br in ['Volkswagen','Audi','BMW','Mercedes-Benz','Smart','Opel','Porshe']:
        data['car_brand_country'][i] = 'Germany'
    elif car_br in ['Mazda','Honda','Suzuki','Mitsubishi','Nissan','Toyota','Lexus','Subaru','Daihatsu']:
        data['car_brand_country'][i] = 'Japan'
    elif car_br in ['Kia','Hyundai']:
        data['car_brand_country'][i] = 'Korea'
    elif car_br in ['Renault','Peugeot','Citroen']:
        data['car_brand_country'][i] = 'France'
    elif car_br in ['Ford','Chevrolet','Jeep','Dodge','Chrysler']:
        data['car_brand_country'][i] = 'USA'
    elif car_br in ['Fiat','AlfaRomeo','Lancia']:
        data['car_brand_country'][i] = 'Italy'
    elif car_br in ['LandRover','Jaguar','Mini']:
        data['car_brand_country'][i] = 'UK'
    elif car_br in ['Volvo','Saab']:
        data['car_brand_country'][i] = 'Sweden'
    else:
        data['car_brand_country'][i] = 'Other'

    for j in freq_car_brand.index:
        if j == car_br and freq_car_brand[j] < data_len_treshold:
            data['car_brand'][i] = 'Other'
            print('******************************************************')

freq_car_brand = data['car_brand'].value_counts()
freq_car_brand

sns.catplot(x='car_brand',
            data=data,
            kind='count')

plt.show()

## Lets see a new variable

sns.catplot(x='car_brand_country',
            data=data,
            kind='count')

plt.show()

## location

freq_loc = data['location'].value_counts()
freq_loc
data_loc_threshold = len(data)/100

for i in range(0,len(data)):
    print(i)
    car_loc = data['location'][i]
    for j in freq_loc.index:
        if j == car_loc and freq_loc[j] < data_loc_threshold:
            data['location'][i] = 'Other'
            print('******************************************************')

freq_car_loc = data['location'].value_counts()
freq_car_loc

sns.catplot(x='location',
            data=data,
            kind='count')

plt.show()

## Type - it looks ok for now

sns.catplot(x='car_type',
            data=data,
            kind='count')

plt.show()

# freq_typ = data['car_type'].value_counts()
# freq_typ
# data_typ_threshold = len(data)/100

# for i in range(0,len(data)):
#     print(i)
#     car_typ = data['car_type'][i]
#     for j in freq_typ.index:
#         if j == car_typ and freq_typ[j] < data_typ_threshold:
#             data['car_type'][i] = 'Other'
#             print('******************************************************')

# freq_car_typ = data['car_type'].value_counts()
# freq_car_typ


## car_gas_type - leave for now

sns.catplot(x='car_gas_type',
            data=data,
            kind='count')

plt.show()

## car_colour - można coś zmienić ale póki co zostawiam

sns.catplot(x='car_colour',
            data=data,
            kind='count')

plt.show()

## car_technical_condition - imbalanced - better to delete this variable (?) - it may change price a lot on the other hand

sns.catplot(x='car_technical_condition',
            data=data,
            kind='count')

plt.show()

## car_gears_type - looks good

sns.catplot(x='car_gears_type',
            data=data,
            kind='count')

plt.show()

## car_country - 3 zmienne w tym jedna przeważająca ok 60% druga około 10% a trzecia około 30% /looks ok 

sns.catplot(x='car_country',
            data=data,
            kind='count')

plt.show()

## car_model - zmienna do pominięcia potencjalnie / w vertex ai duża zmienność - pewnie da się ją jakoś wykorzystać / rether delete this var

freq_model = data['car_model'].value_counts()
freq_model

## Now let us again check the data

data.shape
data.describe()
data.corr()

## Check for multicolinearity as year and mileage can be colinear- https://www.statology.org/how-to-calculate-vif-in-python/

y, X = dmatrices('price ~ car_production_year+car_mileage+car_eng_capacity', data=data, return_type='dataframe')

#calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns

#view VIF for each explanatory variable - it seems that VIF for all explanatory variables is below 10, so it means we can carry on with all vars (especially with year and mileage)
vif

data.columns

del data['car_model']


## Text analysis

## Delete stopwords and special signs
pl_stop_words= np.loadtxt("C:\\Users\\justy\\Desktop\\Info\\Inne\\DSC\\UW\\Magisterka\\Polish stopwords.txt", dtype=str, encoding = "UTF-8")
stopwords=set(pl_stop_words)

def remove_stopwords(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array

clean_desc_stop = remove_stopwords(data['offer_description'])

## Remove special characters such as $ , . ! etc.

clean_description=[]
signs = ['opis oferty','.','\\',',','/','!','-',':','(',')','[',']','$','\"','\'','+','–','*']

for i in range(0,len(clean_desc_stop)):
    clean_desc_ascii = clean_desc_stop[i].lower()
    for j in signs:
        if j != '.' or j !='/' or j !='\\'or j !=':'or j !='+'or j !='*':
            clean_desc_ascii = clean_desc_ascii.replace(j,'')
        else:
            clean_desc_ascii = clean_desc_ascii.replace(j,' ')
    clean_description.append(clean_desc_ascii)

data['offer_description_clean']=clean_description
del data['offer_description']

## Convert all to small letters and reindex

data.reset_index(inplace=True)
data.columns

## Lematization - założenie, że jak ktoś pisze o czymś to raczej jest na plus a nie na minus - zasada sprzedaży i marketingu - może badanie ???? -> raczej pozytywne nacechwoanie

import spacy

doc_input = data['offer_description_clean']
dictionary = ['cb', 'radio', 'elektryczny', 'elektrycznie', 'podgrzewany', 'sensor', 'alufelgi','felgi','ubezpieczony', 'ubezpieczenie', 'hak', 'MP3', 'AUX', 'USB', 'CD', 'DVD', 'bluetooth', 'tempomat', 'czujnik', 'opona', 'airbag', 'komputer', 'GPS', 'nawigacja', 'LED', 'halogen', 'ABS', 'ESP', 'aluminium', 'aluminiowy' 'uszkodzony', 'uszkodzenie', 'uszkodzić' 'bezwypadkowy', 'rysa', 'aso', 'garaż', 'podgrzewane', 'podgrzewanie', 'anglik', 'immobiliser', 'wspomaganie', 'zapasowe', 'zima', 'alarm', 'leasing', 'kredyt', 'klimatyzacja', 'skóra', 'skórzać' 'pies', 'szkoda', 'przeciwsłoneczny', 'welurowy', 'bogaty', 'asystent', 'kamera', 'poduszka', 'klimatyzacja', 'vat', 'negocjacja']
dictionary_len = len(dictionary)
temp_lem =pd.DataFrame()

nlp = spacy.load("pl_core_news_sm")

for i in range(0,len(doc_input)):
    print(i)
    doc = nlp(doc_input[i])
    for j in dictionary:
        var = "key_"+str(j)
        temp_lem.at[i,var]=0
    for token in doc:
        if token.lemma_ in dictionary:
            var_nam = "key_"+str(token.lemma_)
            temp_lem.at[i,var_nam]=1

## Let us recode categorical varialbes - hot encoding format

obj_df = data.select_dtypes(include=['object']).copy()
obj_df.head()
del obj_df['location']

obj_df[obj_df.isnull().any(axis=1)]

obj_df_rec = pd.get_dummies(obj_df, columns=["car_type", "car_gas_type", "car_colour", "car_gears_type", "car_country", "car_brand","car_brand_country","car_technical_condition"], prefix=["type", "gas", "colour", "gears", "country", "brand","brand_country","tech_cond"])

df = obj_df_rec
df[['price','car_production_year', 'car_mileage','car_eng_capacity']]=data[['price','car_production_year', 'car_mileage','car_eng_capacity']]

data = df.copy()

data.columns

# append variables to main dataset

for i in dictionary:
    var2 = "key_"+str(i)
    data[var2] = temp_lem[var2]

data.columns

del data['offer_description_clean']

data.dropna()

#data.to_csv('C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/train_data_clean.csv')
data = pd.read_csv('C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/train_data_clean.csv')
del data['Unnamed: 0']

data.dropna()
######################################################################## TRAINING DATASET #############################################################



######################################################################## TEST DATASET #################################################################

data_test

## perform EDA only on train sample to make sure there is no data leakage

data = data_test.copy()

## Check types, shape etc.

data.shape
data.describe()
data.dtypes

## Check correlation - it seems that it is not correlated at all. Why? Probably there are outliers

data.corr()

## Check the data distribution - maybe there are some outliers

ax1 = data.plot.scatter(x='car_production_year',y='price',c='DarkBlue')
plt.show()

# It seems that price have a few big outliers - let us delete them. Let us also delete cars that costs more than 600k, as it seems to be natural barier

std_flag_price = data['price']<=600000
data['std_flag_price']=std_flag_price
data = data[data['std_flag_price']==True]
del data['std_flag_price']

std_flag_price = data['price']>=10000
data['std_flag_price']=std_flag_price
data = data[data['std_flag_price']==True]
del data['std_flag_price']

# Let us check the plot now. Now it is much better

ax1 = data.plot.scatter(x='car_production_year',y='price',c='blue')
plt.show()


# Now let us take a look at car production year. We may leave only cars were produced between 2000 and now

data.car_production_year.value_counts()
data['car_production_year'].hist(bins=30)
plt.show()

std_flag_year = data['car_production_year']>=2000
data['std_flag_year']=std_flag_year
data = data[data['std_flag_year']==True]
del data['std_flag_year']

ax1 = data.plot.scatter(x='car_production_year',y='price',c='DarkBlue')
plt.show()

# We can see that the pattern is more like polynomial order 2 not linear

# Now let us take a look at car mileage. There are some outliers. Let us then delete them as we did in car price

ax2 = data.plot.scatter(x='car_mileage',y='price',c='DarkBlue')
plt.show()

# Let us get rid of cars that have mileage over 500k - it is very unusual and we don`t want to look at such cars. Now it looks good.

std_flag_mil = (data['car_mileage'])<500000
data['std_flag_mil']=std_flag_mil
data = data[data['std_flag_mil']==True]
del data['std_flag_mil']

ax2 = data.plot.scatter(x='car_mileage',y='price',c='DarkBlue')
plt.show()

# Now let us look at end capacity. It seems that there are some nonsense data. Let us get rid of the cars that have eng cap below 1k cm^3 over 6k cm^3

ax3 = data.plot.scatter(x='car_eng_capacity',y='price',c='DarkBlue')
plt.show()

std_flag_cap = (data['car_eng_capacity'])>=1000
data['std_flag_cap']=std_flag_cap
data = data[data['std_flag_cap']==True]
del data['std_flag_cap']

std_flag_cap = (data['car_eng_capacity'])<=6000
data['std_flag_cap']=std_flag_cap
data = data[data['std_flag_cap']==True]
del data['std_flag_cap']

ax3 = data.plot.scatter(x='car_eng_capacity',y='price',c='DarkBlue')
plt.show()

data = data.reset_index(drop=True)

## Categorical data inspection

## car brand

sns.catplot(x='car_brand',
            data=data,
            kind='count')

plt.show()

freq_car_brand = data['car_brand'].value_counts()
freq_car_brand
data_len_treshold = int(len(data_test)/100)

len(freq_car_brand)
data['car_brand_country']=''

for i in range(0,len(data)):
    print(i)
    car_br = data['car_brand'][i]
    if car_br in ['Volkswagen','Audi','BMW','Mercedes-Benz','Smart','Opel','Porshe']:
        data['car_brand_country'][i] = 'Germany'
    elif car_br in ['Mazda','Honda','Suzuki','Mitsubishi','Nissan','Toyota','Lexus','Subaru','Daihatsu']:
        data['car_brand_country'][i] = 'Japan'
    elif car_br in ['Kia','Hyundai']:
        data['car_brand_country'][i] = 'Korea'
    elif car_br in ['Renault','Peugeot','Citroen']:
        data['car_brand_country'][i] = 'France'
    elif car_br in ['Ford','Chevrolet','Jeep','Dodge','Chrysler']:
        data['car_brand_country'][i] = 'USA'
    elif car_br in ['Fiat','AlfaRomeo','Lancia']:
        data['car_brand_country'][i] = 'Italy'
    elif car_br in ['LandRover','Jaguar','Mini']:
        data['car_brand_country'][i] = 'UK'
    elif car_br in ['Volvo','Saab']:
        data['car_brand_country'][i] = 'Sweden'
    else:
        data['car_brand_country'][i] = 'Other'

    for j in freq_car_brand.index:
        if j == car_br and freq_car_brand[j] < data_len_treshold:
            data['car_brand'][i] = 'Other'
            print('******************************************************')

freq_car_brand = data['car_brand'].value_counts()
freq_car_brand

sns.catplot(x='car_brand',
            data=data,
            kind='count')

plt.show()

## Lets see a new variable

sns.catplot(x='car_brand_country',
            data=data,
            kind='count')

plt.show()

## location

freq_loc = data['location'].value_counts()
freq_loc
data_loc_threshold = len(data)/100

for i in range(0,len(data)):
    print(i)
    car_loc = data['location'][i]
    for j in freq_loc.index:
        if j == car_loc and freq_loc[j] < data_loc_threshold:
            data['location'][i] = 'Other'
            print('******************************************************')

freq_car_loc = data['location'].value_counts()
freq_car_loc

sns.catplot(x='location',
            data=data,
            kind='count')

plt.show()

## Type - it looks ok for now

sns.catplot(x='car_type',
            data=data,
            kind='count')

plt.show()

# freq_typ = data['car_type'].value_counts()
# freq_typ
# data_typ_threshold = len(data)/100

# for i in range(0,len(data)):
#     print(i)
#     car_typ = data['car_type'][i]
#     for j in freq_typ.index:
#         if j == car_typ and freq_typ[j] < data_typ_threshold:
#             data['car_type'][i] = 'Other'
#             print('******************************************************')

# freq_car_typ = data['car_type'].value_counts()
# freq_car_typ


## car_gas_type - leave for now

sns.catplot(x='car_gas_type',
            data=data,
            kind='count')

plt.show()

## car_colour - można coś zmienić ale póki co zostawiam

sns.catplot(x='car_colour',
            data=data,
            kind='count')

plt.show()

## car_technical_condition - imbalanced - better to delete this variable (?) - it may change price a lot on the other hand

sns.catplot(x='car_technical_condition',
            data=data,
            kind='count')

plt.show()

## car_gears_type - looks good

sns.catplot(x='car_gears_type',
            data=data,
            kind='count')

plt.show()

## car_country - 3 zmienne w tym jedna przeważająca ok 60% druga około 10% a trzecia około 30% /looks ok 

sns.catplot(x='car_country',
            data=data,
            kind='count')

plt.show()

## car_model - zmienna do pominięcia potencjalnie / w vertex ai duża zmienność - pewnie da się ją jakoś wykorzystać / rether delete this var

freq_model = data['car_model'].value_counts()
freq_model

## Now let us again check the data

data.shape
data.describe()
data.corr()

## Check for multicolinearity as year and mileage can be colinear- https://www.statology.org/how-to-calculate-vif-in-python/

y, X = dmatrices('price ~ car_production_year+car_mileage+car_eng_capacity', data=data, return_type='dataframe')

#calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns

#view VIF for each explanatory variable - it seems that VIF for all explanatory variables is below 10, so it means we can carry on with all vars (especially with year and mileage)
vif

data.columns

del data['car_model']


## Text analysis

## Delete stopwords and special signs
pl_stop_words= np.loadtxt("C:\\Users\\justy\\Desktop\\Info\\Inne\\DSC\\UW\\Magisterka\\Polish stopwords.txt", dtype=str, encoding = "UTF-8")
stopwords=set(pl_stop_words)

def remove_stopwords(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array

clean_desc_stop = remove_stopwords(data['offer_description'])

## Remove special characters such as $ , . ! etc.

clean_description=[]
signs = ['opis oferty','.','\\',',','/','!','-',':','(',')','[',']','$','\"','\'','+','–','*']

for i in range(0,len(clean_desc_stop)):
    clean_desc_ascii = clean_desc_stop[i].lower()
    for j in signs:
        if j != '.' or j !='/' or j !='\\'or j !=':'or j !='+'or j !='*':
            clean_desc_ascii = clean_desc_ascii.replace(j,'')
        else:
            clean_desc_ascii = clean_desc_ascii.replace(j,' ')
    clean_description.append(clean_desc_ascii)

data['offer_description_clean']=clean_description
del data['offer_description']

## Convert all to small letters and reindex

data.reset_index(inplace=True)
data.columns

## Lematization - założenie, że jak ktoś pisze o czymś to raczej jest na plus a nie na minus - zasada sprzedaży i marketingu - może badanie ???? -> raczej pozytywne nacechwoanie

import spacy

doc_input = data['offer_description_clean']
dictionary = ['cb', 'radio', 'elektryczny', 'elektrycznie', 'podgrzewany', 'sensor', 'alufelgi','felgi','ubezpieczony', 'ubezpieczenie', 'hak', 'MP3', 'AUX', 'USB', 'CD', 'DVD', 'bluetooth', 'tempomat', 'czujnik', 'opona', 'airbag', 'komputer', 'GPS', 'nawigacja', 'LED', 'halogen', 'ABS', 'ESP', 'aluminium', 'aluminiowy' 'uszkodzony', 'uszkodzenie', 'uszkodzić' 'bezwypadkowy', 'rysa', 'aso', 'garaż', 'podgrzewane', 'podgrzewanie', 'anglik', 'immobiliser', 'wspomaganie', 'zapasowe', 'zima', 'alarm', 'leasing', 'kredyt', 'klimatyzacja', 'skóra', 'skórzać' 'pies', 'szkoda', 'przeciwsłoneczny', 'welurowy', 'bogaty', 'asystent', 'kamera', 'poduszka', 'klimatyzacja', 'vat', 'negocjacja']
dictionary_len = len(dictionary)
temp_lem =pd.DataFrame()

nlp = spacy.load("pl_core_news_sm")

for i in range(0,len(doc_input)):
    print(i)
    doc = nlp(doc_input[i])
    for j in dictionary:
        var = "key_"+str(j)
        temp_lem.at[i,var]=0
    for token in doc:
        if token.lemma_ in dictionary:
            var_nam = "key_"+str(token.lemma_)
            temp_lem.at[i,var_nam]=1

## Let us recode categorical varialbes - hot encoding format

obj_df = data.select_dtypes(include=['object']).copy()
obj_df.head()
del obj_df['location']

obj_df[obj_df.isnull().any(axis=1)]

obj_df_rec = pd.get_dummies(obj_df, columns=["car_type", "car_gas_type", "car_colour", "car_gears_type", "car_country", "car_brand","car_brand_country","car_technical_condition"], prefix=["type", "gas", "colour", "gears", "country", "brand","brand_country","tech_cond"])

df = obj_df_rec
df[['price','car_production_year', 'car_mileage','car_eng_capacity']]=data[['price','car_production_year', 'car_mileage','car_eng_capacity']]

data = df.copy()

data.columns

# append variables to main dataset

for i in dictionary:
    var2 = "key_"+str(i)
    data[var2] = temp_lem[var2]

data.columns

del data['offer_description_clean']

data.dropna()

#data.to_csv('C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/test_data_clean.csv')
data = pd.read_csv('C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/test_data_clean.csv')
del data['Unnamed: 0']

data.dropna()

######################################################################## TEST DATASET #################################################################



######################################################################## MODELING #####################################################################

## Load data
data_train = pd.read_csv('C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/train_data_clean.csv')
del data_train['Unnamed: 0']
data_test = pd.read_csv('C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/test_data_clean.csv')
del data_test['Unnamed: 0']

## Divide to X and y

X_train = pd.DataFrame(data_train.drop(['price'],axis=1))
y_train = pd.DataFrame(data_train['price'])

X_test = pd.DataFrame(data_test.drop(['price'],axis=1))
y_test = pd.DataFrame(data_test['price'])

## Scale train, save parameters for scaling and use same parameters for test to scale

scaler = preprocessing.MinMaxScaler()
data_train[["car_production_year", "car_mileage", "car_eng_capacity", "price"]] = scaler.fit_transform(data_train[["car_production_year", "car_mileage", "car_eng_capacity", "price"]])

import joblib
joblib.dump(scaler, 'C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/minmax_scaler.pkl')

scaler = joblib.load('C:/Users/justy/Desktop/Info/Inne/DSC/UW/Semestr III/ML2/Projects/Regression/minmax_scaler.pkl')
data_test[["car_production_year", "car_mileage", "car_eng_capacity", "price"]] = scaler.transform(data_test[["car_production_year", "car_mileage", "car_eng_capacity", "price"]])

y_test.shape
## Make sure that train and test data have same number of variables - after encoding there might be a few missing parts in test

X_train, X_test = X_train.align(X_test, join='left', axis=1)
X_test.fillna(0, inplace=True)

## Simple Linear regression

regLin1 = LinearRegression()
regLin1.fit(X_train, y_train)
y_pred = regLin1.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['price'])
regLin1.score(X_test, y_test)

y_test.shape
y_test.dtypes
y_pred.shape
y_pred = y_pred.values

y_pred_original = scaler.inverse_transform(y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_original))

metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
metrics.mean_absolute_percentage_error(y_test, y_pred)
relative_root_mean_squared_error(np.array(y_test), y_pred)

# # get importance
# importance = regLin1.coef_
# # summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()


## Polynomial - features

poly = PolynomialFeatures()
poly.fit_transform(X_train)

print(poly)

regPoly = LinearRegression()
regPoly.fit(poly, y_train)
y_pred = regPoly.predict(X_test)
regPoly.score(poly, y_test)

metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## RIDGE regression

regRidge = Ridge().fit(X_train, y_train)
y_pred = regRidge.predict(X_test)
regRidge.score(X_test, y_test)

metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## RIDGECV regression

regRidgeCV = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)
y_pred = regRidgeCV.predict(X_test)
regRidgeCV.score(X_test, y_test)

metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

## Elastic NET regression

regrElastic = ElasticNet(random_state=0)
y_pred = regrElastic.predict(X_test)
regrElastic.score(X_test, y_test)

metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## Random Forest Regression - https://github.com/mk-gurucharan/Regression/blob/master/Models/Random_Forest_Regression.ipynb / https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74


## Hyperparameter tuning

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 60, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [20, 50, 1000]
# Minimum number of samples required at each leaf node
min_samples_leaf = [100, 200, 400]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
regRF = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = regRF, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = 3)

rf_random.fit(X_train, y_train)

rf_random.best_params_

# {'n_estimators': 1800, 'min_samples_split': 50, 'min_samples_leaf': 100, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': True}

regRF = RandomForestRegressor(n_estimators = 1800, min_samples_split = 50, min_samples_leaf= 100, max_features = "auto", max_depth= 40, bootstrap= True, random_state = 0)
regRF.fit(X_train, y_train)

y_pred = regRF.predict(X_test)
regRF.score(X_test, y_test)

metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
relative_root_mean_squared_error(np.array(y_test), y_pred)
relative_mae(y_test['price'], y_pred)
metrics.mean_absolute_percentage_error(y_test, y_pred)


## Gradient Boosting

regGB = sklearn.ensemble.GradientBoostingRegressor(random_state=0)
regGB.fit(X_train, y_train)
y_pred =regGB.predict(X_train)
y_pred =regGB.predict(X_test)
regGB.score(X_test, y_test)
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## Neural Network

regNN = sklearn.neural_network.MLPRegressor(random_state=0, max_iter=50, alpha = 0.05, hidden_layer_sizes = 250, activation = 'relu', learning_rate = 'constant').fit(X_train, y_train)
y_pred = regNN.predict(X_test)
regNN.score(X_test, y_test)
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
mean(y_test['price'])
relative_root_mean_squared_error(y_test['price'], y_pred)
relative_mae(y_test['price'], y_pred)




## Test Neural Network - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

estimator = MLPRegressor()


param_grid = {'hidden_layer_sizes': [10,100,250,500],
          'activation': ['relu','tanh','logistic'],
          'alpha': [0.0001,0.001, 0.05, 0.1],
          'learning_rate': ['constant','adaptive'],
          'solver': ['adam']}

gsc = GridSearchCV(
    estimator,
    param_grid,
    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=4)

grid_result = gsc.fit(X, y)


best_params = grid_result.best_params_

## {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': 250, 'learning_rate': 'constant', 'solver': 'adam'} - best params for 30k obs recoded from -1 to 1

best_mlp = MLPRegressor(hidden_layer_sizes = best_params["hidden_layer_sizes"], 
                        activation =best_params["activation"],
                        solver=best_params["solver"],
                        max_iter= 200, n_iter_no_change = 200
              )

scoring = {
           'abs_error': 'neg_mean_absolute_error',
           'squared_error': 'neg_mean_squared_error',
           'r2':'r2'}

scores = cross_validate(best_mlp, X, y, cv=10, scoring=scoring, return_train_score=True, return_estimator = True)

##END TEST


## SVM - https://github.com/mk-gurucharan/Regression/blob/master/Models/Support_Vector_Regression.ipynb

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred

metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Visualising the SVR results (for higher resolution and smoother curve)

plt.scatter(X_test['car_mileage'], y_test, color = 'red')
plt.scatter(X_test['car_mileage'], y_pred, color = 'green')
plt.title('SVR Regression')
plt.xlabel('Car mileage')
plt.ylabel('Price')
plt.show()


## nieistotne zmienne w modelu / brakiem skalowania wartości numerycznych / sprawdzanie rozkładu normalnego itp. / 
## Testy modelu standardowego pod względem ekonometrycznym / statystycznym / raczej lepszy będzie model polynomial 2

from sklearn.feature_selection import RFECV

rfe = RFECV(regNN, step=0.05)
fit = rfe.fit(X, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))



## Test for features selection

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification


svc = SVC(kernel="linear")
# The "accuracy" scoring shows the proportion of correct classifications

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(
    estimator=svc,
    step=1,
    cv=StratifiedKFold(2),
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
    rfecv.grid_scores_,
)
plt.show()



## Test feature importance
from sklearn.feature_selection import SelectFromModel

fs = SelectFromModel(regRF)
fs.fit(X_train, y_train)

X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


## Test feature importance from - https://github.com/WillKoehrsen/Data-Analysis/blob/master/random_forest_explained/Random%20Forest%20Explained.ipynb

# Get numerical feature importances
importances = list(regRF.feature_importances_)
feature_list = list(X.columns)


# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances

important_x_feat=[]

for i in feature_importances:
    if i[1] >=0.005:
        important_x_feat.append(i[0])

X=X.get(['car_production_year', 'car_eng_capacity', 'car_mileage', 'country_Sprowadzany niezarejestrowany ', 'type_SUV ', 'gas_Benzyna ', 'gears_Automatyczna ', 'gears_Manualna ', 'country_z Polski '])


# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances'); 

plt.show()

## Test rescaling output

from sklearn.compose import TransformedTargetRegressor

regNN = sklearn.neural_network.MLPRegressor(random_state=0, max_iter=50, alpha = 0.05, hidden_layer_sizes = 250, activation = 'relu', learning_rate = 'constant')
scaler = preprocessing.MinMaxScaler()

tt = TransformedTargetRegressor(regressor=regNN,
                                  transformer = scaler).fit(X_train, y_train)

y_pred = tt.predict(X_test)

s=scaler.fit(X_test)

tt.score(X_test, y_test)

inv_y_pred = s.inverse_transform(y_pred.reshape(-1, 1))
inv_y_test = qt_y.inverse_transform(y_test.reshape(-1, 1))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

## Bład -> przed skalowaniem trzeba podzielić dataset na train i test i skalowanie robić tylko na trainie (aktualnie wyciek danych) / do reskalowania testu użyc tych samych parametrów std i mean jak w przypadku traina 






## Way to compare different models

def relative_root_mean_squared_error(true, pred):
    num = np.sqrt(metrics.mean_squared_error(true, pred))
    rrmse_loss = num/mean(true)
    return rrmse_loss

def relative_mae(true, pred):
    num = metrics.mean_absolute_error(true, pred)
    rrmse_loss = num/(max(true) - min (true))
    return rrmse_loss
