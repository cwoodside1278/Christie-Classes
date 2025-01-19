#Christie Woodside Homework 2
#Sept 3, 2024

import pandas as pd
#from IPython.display import display
import requests
import wbgapi as wb
#import numpy as np


'''Homework part 1 from Blackboard directions'''
# to check for redundent info in the columns
def redundant_check(df):
    r = {}
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            #checking to see if they equal eachother
            col1 = df[columns[i]]
            col2 = df[columns[j]]
            #print(col1)
            #print(col2)
            if col1.equals(col2):
                r[columns[i]] = columns[j]

            #checking to see if the same values or data type
            elif col1.dtype == col2.dtype and col1.nunique() == col2.nunique():
                unique1 = set(col1.dropna())
                unique2 = set(col2.dropna())
                if unique1 == unique2:
                    r[columns[i]] = columns[j]
            #print(r)
    return r


'''_______________From question 1__________________________________'''
df1 = pd.read_csv('PLACES__Local_Data_for_Better_Health__County_Data_2024_release_20240828.csv')
print(df1)
print("\ndata types for each column in PLACES dataset:")
print(df1.dtypes)

rc = redundant_check(df1)
print("\n redundent info in PLACES dataset:")
print(rc if rc 
      else "no redundent information")

print("\n unique data for each column in PLACES dataset:")
for c in df1.columns:
    print(f"\n unique values in '{c}':")
    print(df1[c].unique())

print("\n descriptives for each column in PLACES dataset:")
print(df1.describe())





'''______________________from question 2________________________________________'''
api = 'mauvecrane23'
base = 'https://aqs.epa.gov/data/api/'
endpoint = 'dailyData/byCounty'
#state_codes = [f'{i:02}' for i in range(1, 58) if i not in [3, 7]] #want all the states
# code list so helpful :https://www.epa.gov/aqs/aqs-code-list
#this one is so better: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www3.epa.gov/ttnairs1/airsaqsORIG/manuals/city_names.pdf
ptemp = {
    'email': 'christie.woodside@gwu.edu', 
    'key': api,
    'param': "88101",
    'state': '01', #alabama
    'county': '097', #is Mobile #010000 should be all counties https://unicede.air-worldwide.com/unicede/unicede_us_fips_codes.html
}
year = ['2019', '2020', '2021']
for y in year:
    ptemp['bdate'] = f'{y}0101'  
    ptemp['edate'] = f'{y}1231' 

combine = [] 
response = requests.get(f"{base}{endpoint}", params=ptemp)
if response.status_code == 200:
    data = response.json()
    if 'Data' in data:
        combine.extend(data['Data'])
    else:
        print('uh oh')

#to get the combined dataset information
if combine:
    d2 = pd.DataFrame(combine)
    print("\ndata types in columns for AQS dataset:")
    print(d2.dtypes)

    ra = redundant_check(d2)
    # print("\n redundent info in AQS dataset:")
    # print(ra if ra
    #   else "no redundent information")
    print("\nRedundant information in AQS dataset:")

    if ra:
        print(ra)
        for col1, col2 in ra.items():
            print(f"\nComparing values in '{col1}' and '{col2}':")
            print(f"First 10 values in '{col1}':")
            print(d2[col1].head(10))
            print(f"\nFirst 10 values in '{col2}':")
            print(d2[col2].head(10))
    else:
        print("No redundant information")

    print("\n unique data in columns for AQS dataset:")
    for c in d2.columns:
        print(f"\n unique values in '{c}':")
        print(d2[c].unique())
        
    print("\n descriptives for each column in AQS dataset:")
    print(d2.describe())



'''_____________________________from question 3_______________________________________'''
i_code = ['NY.GDP.MKTP.PP.KD', 'EN.ATM.CO2E.KT']
eu_countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden"
]
iso_codes = []
for country in eu_countries:
    lookup = wb.economy.coder(country)
    if lookup:
        iso_codes.append(lookup)

data = wb.data.DataFrame(i_code, iso_codes, range(1994,2012))

data.to_csv('EU_WBD.csv', index=True)
print(data)
print("\n data types in columns for WB dataset:")
print(data.dtypes)

rw = redundant_check(data)
print("\n redundent info in WB dataset:")
print(rw if rw
    else "no redundent information")

# print("\nRedundant information in WB dataset:")
# if rw:
#     print(rw)
#     for col1, col2 in rw.items():
#         print(f"\nComparing values in '{col1}' and '{col2}':")
#         print(f"First 10 values in '{col1}':")
#         print(d2[col1].head(10))
#         print(f"\nFirst 10 values in '{col2}':")
#         print(d2[col2].head(10))
# else:
#     print("No redundant information")

print("\n unique data in columns for WB dataset:")
for c in data.columns:
        print(f"\n unique values in '{c}':")
        print(data[c].unique())
        
print("\n descriptives for each column in World bank dataset:")
print(data.describe())
