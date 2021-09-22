import pandas as pd

#%%

DF=pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
print(DF)

#%%

SR=pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
print(SR)

#%%

datas=pd.read_csv(r"/Users/arjunkapoor/Desktop/exp 2pattern csv.csv")
print(datas.shape)

#%%

print(datas.country)
print(datas.country[0])



#%%

print(datas.loc[56, 'country'])

#%%

datas["points"]= "100"
print(datas["points"])

#%%

print(datas.head())
print(datas.points.mean())


#%%

print(datas.groupby(['country']).price.agg([len, min, max]))

#%%

countries_reviewed = datas.groupby(['country', 'province']).description.agg([len])
mi = countries_reviewed.index
print(type(mi))

#%%

countries_reviewed = countries_reviewed.reset_index()
print(countries_reviewed.sort_values(by='len'))

#%%

print(datas.dtypes)

#%%

datas.points.astype('float64')
print(datas.points.dtype)

#%%

print(datas[pd.isnull(datas.country)])
print(datas.region_2.fillna("Unknown"))

#%%

print(datas.rename(columns={'points': 'score'}))

#%%

print(datas.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns'))

#%%
