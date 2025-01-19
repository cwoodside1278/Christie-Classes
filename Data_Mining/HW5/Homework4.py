# %%
# import required libraries
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import wbdata
import pandas as pd
import wbgapi as wb
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder

# %%
# CATEGORICAL DATA PART
#
# %%
# Problem 1: Basic Label Encoding
# Given a list of categorical values ['low', 'medium', 'high', 'low'], encode them using label encoding.

categories = ['low', 'medium', 'high', 'low']

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(categories)

encoded_labels  

# %%
#Problem 2: One-Hot Encoding with Pandas
#Convert a list of categorical values ['red', 'blue', 'green', 'blue'] into one-hot encoded variables using pandas.

categories = ['red', 'blue', 'green', 'blue']

df = pd.get_dummies(categories)

df 
# %%
#Problem 3: Filling Missing Categorical Values
#Given a column with categorical values ['A', 'B', 'A', None, 'C', 'A', None], fill missing values with the mode (most frequent value).

categories = ['A', 'B', 'A', None, 'C', 'A', None]

df = pd.DataFrame(categories, columns=['Category'])

df['Category'].fillna(df['Category'].mode()[0], inplace=True)

df['Category'] 

# %%
#Problem 4: Frequency Encoding
#Replace the categorical values ['A', 'B', 'A', 'C', 'B', 'A'] with their frequency counts.
categories = ['A', 'B', 'A', 'C', 'B', 'A']

df = pd.DataFrame(categories, columns=['Category'])

frequency_encoding = df['Category'].map(df['Category'].value_counts())

frequency_encoding

# %%
#Problem 5: Combining Rare Categories
#Given a list of categorical values ['A', 'B', 'A', 'C', 'D', 'A', 'D', 'C'], combine categories that occur fewer than 2 times into a new category called 'Other'.

categories = ['A', 'B', 'A', 'C', 'D', 'A', 'D', 'C']

df = pd.DataFrame(categories, columns=['Category'])

threshold = 2
value_count = df['Category'].value_counts()
df['Category'] = df['Category'].apply(lambda x: x if value_count[x] >= threshold else 'Other')

df['Category'] 

# %%
#Problem 6: Inverse Label Encoding
#Given a list of encoded values [0, 1, 2, 0], decode it back into ['A', 'B', 'C', 'A'].

encoded_labels = [0, 1, 2, 0]

categories = ['A', 'B', 'C']

encoder = LabelEncoder()
encoder.fit(categories)

decoded_labels = encoder.inverse_transform(encoded_labels)

decoded_labels 

# %%
#Problem 7: Mapping Categories
#Given a column of values ['small', 'medium', 'large'], map them to ['S', 'M', 'L'] using a 

categories = ['small', 'medium', 'large']

mapping = {'small': 'S', 'medium': 'M', 'large': 'L'}

mapped_categories = [mapping[item] for item in categories]

mapped_categories 

# %%
#Problem 8: Ordinal Encoding
#Encode the categorical list ['cold', 'warm', 'hot'] into ordinal values, where 'cold' < 'warm' < 'hot'.

categories = ['cold', 'warm', 'hot']

ordinal_mapping = {'cold': 0, 'warm': 1, 'hot': 2}

ordinal_encoded = [ordinal_mapping[item] for item in categories]

ordinal_encoded

# %%




# %%
# World Bank Problem part


# Problem 1 We are continuing to work with World Bank data. Use the dataset from the previous homework. Add the 'Income Level' variable. HINT, here is the code how to get Income Level for all countries
# Fetch country metadata using the World Bank API
#countries = wbdata.get_countries()
# Create a DataFrame for country metadata
#country_metadata = pd.DataFrame({
#    'countryCode': [c['id'] for c in countries],
#    'country': [c['name'] for c in countries],
#    'incomeLevel': [c['incomeLevel']['value'] for c in countries]
#})

#Remove the countries where the 'Income Level' is 'Aggregates', NaN, or 'Not classified', which means you will only keep 'Low income', 'Upper middle income', 'High income', and 'Lower middle income'. Use this new categorical variable and apply every categorical encoding method discussed in class. For the target encoding use GDP. 



# %%
indicator_dict = {
    "EN.ATM.CO2E.PC": "CO2 emissions",
    "NY.GDP.PCAP.PP.KD": "GDP per capita, PPP",
    "GB.XPD.RSDV.GD.ZS": "Research and development expenditure",
    "NE.TRD.GNFS.ZS": "Trade",
    "EG.FEC.RNEW.ZS": "Renewable energy consumption",
    "TX.VAL.TECH.MF.ZS": "Medium and high-tech industry",
    "SP.URB.TOTL.IN.ZS": "Urban population",
    "EG.USE.PCAP.KG.OE": "Energy use"
}

raw_data = wb.data.DataFrame(list(indicator_dict.keys()))

cleaned_data = raw_data.rename(columns=lambda x: x.lstrip("YR") if x.startswith("YR") else x)
cleaned_data = cleaned_data.rename(index=indicator_dict)

cleaned_data.index.set_names(["Country", "Indicator"], inplace=True)
cleaned_data.columns = pd.to_numeric(cleaned_data.columns)
cleaned_data.columns.name = "Year"

filtered_data = cleaned_data.loc[:, 1990:2023]
filtered_data.to_csv("wb_data.csv")

# %%
data = pd.read_csv('wb_data.csv')

# %%
countries = wbdata.get_countries()

country_metadata = pd.DataFrame({
    'countryCode': [c['id'] for c in countries],
    'country': [c['name'] for c in countries],
    'incomeLevel': [c['incomeLevel']['value'] for c in countries]
})
# %%

levels = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
clean_country__metadata = country_metadata[country_metadata['incomeLevel'].isin(levels)]
merged_data = pd.merge(data, clean_country__metadata, left_on='Country', right_on='countryCode', how='inner')


# %%
# One Hot Encoding
one_hot_data = pd.get_dummies(merged_data, columns=['incomeLevel'])
one_hot_data.head()

# %%

# 
# Label Encoding 

label_encoder = LabelEncoder()
merged_data['incomeLevel_encoded'] = label_encoder.fit_transform(merged_data['incomeLevel'])
merged_data.head()

# %%

# Target Encoding

gdp_data= merged_data[merged_data['Indicator'] == 'GDP per capita, PPP'][['Country', '2023']]
gdp_data = gdp_data.rename(columns={'2023': 'GDP per capita, PPP'})

merged_data = pd.merge(merged_data, gdp_data, on='Country', how='left')
merged_data['incomeLevel_target_encoded'] = merged_data.groupby('incomeLevel')['GDP per capita, PPP'].transform('mean')
merged_data[['incomeLevel', 'GDP per capita, PPP', 'incomeLevel_target_encoded']].head()


# %%
# Binary encoding

binary_encoder = ce.BinaryEncoder(cols=['incomeLevel'])
binary_data = binary_encoder.fit_transform(merged_data)
binary_data.head()

# %%

# Frequency Encoding

frequency_map = merged_data['incomeLevel'].value_counts(normalize=True).to_dict()
merged_data['incomeLevel_frequency'] = merged_data['incomeLevel'].map(frequency_map)

merged_data['incomeLevel_frequency'].head()

# %%

# Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
merged_data['incomeLevel_ordinal'] = ordinal_encoder.fit_transform(merged_data[['incomeLevel']])
merged_data.head()


# %%
# Problem 2: Using the data from the previous problem, remove the countries where the ‘Income Level’ is ‘Aggregates’, NaN, or ‘Not classified’. 
# This means you will only keep ‘Low income’, ‘Upper middle income’, ‘High income’, and ‘Lower middle income’. For CO2 variable, please impute missing values using all but multivariate imputations discussed in class

# %%

income_levels = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
final_data = merged_data[merged_data['incomeLevel'].isin(income_levels)]

year_columns = final_data.loc[:, '1990':'2023'].columns

# %%

# Mean
final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns] = final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns].apply(lambda x: x.fillna(x.mean()), axis=0)

# %%

# Meadina
final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns] = final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns].apply(lambda x: x.fillna(x.median()), axis=0)

# %%

# Mode
final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns] = final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns].apply(lambda x: x.fillna(x.mode()), axis=0)

# %%

# ffill
final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns] = final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns].fillna(method='ffill', axis=0)

# %%

# bfill
final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns] = final_data.loc[final_data['Indicator'] == 'CO2 emissions', year_columns].fillna(method='bfill', axis=0)

# %%
final_data.head()

# %%
