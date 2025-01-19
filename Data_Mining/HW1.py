#Homework 1 Data Mining
#Christie Woodside

import pandas as pd
from IPython.display import display
import requests
import wbgapi as wb

'''Question #1'''
#Import csv---------------------------------
#df = pd.read_csv('500_Cities__Local_Data_for_Better_Health__2019_release_20240828.csv', na_values=['NA', 'missing'])
df = pd.read_csv('PLACES__Local_Data_for_Better_Health__County_Data_2024_release_20240828.csv')

#Adding the details ---------------------
detail = {
    'Year': ["year", 'year', 'Number'],
    'StateAbbr': ['State abbreviation', "stateabbr", 'Text'],
    'StateDesc' :['State name', 'statedesc', 'Text'],
    'LocationName' : ['County name', 'locationname', 'Text'],
    'DataSource' : ['Data source', 'datasource', 'Text'],
    'Category': ['Topic', 'category', 'Text'],
    'Measure' : ['Measure full name', 'measure', 'Text'],
    'Data_Value_Unit' : ['The data value unit, such as "%" for percentage', 'data_value_unit', 'Text'],
    'Data_Value_Type' : ['The data type, such as age-adjusted prevalence or crude prevalence', 'data_value_type', 'Text'],
    'Data_Value' : ['Data Value, such as 14.7', 'data_value', 'Number'],
    'Data_Value_Footnote_Symbol' :["Footnote symbol", 'data_value_footnote_symbol', 'Text'],
    'Data_Value_Footnote': ['Footnote text', 'data_value_footnote', 'Text'],
    'Low_Confidence_Limit' : ['Low confidence limit', 'low_confidence_limit','Number'],
    'High_Confidence_Limit' : ['High confidence limit', 'high_confidence_limit', 'Number'],
    'TotalPopulation' :["Total population of census 2022 estimates", 'totalpopulation', 'Number']
}

# Turn dictionary to a DataFrame
deet = pd.DataFrame.from_dict(detail, orient='index', columns=['Description:', 'API Field Name:', 'Data Type:'])

# Display the DataFrame
print("Details and Information Table for the Dataset")
display(deet)
print(f" \n This dataset contains model-based county estimates. PLACES covers the entire United States—50 states and the District of Columbia—at county, place, census tract, and ZIP Code Tabulation Area levels. It provides information uniformly on this large scale for local areas at four geographic levels. Estimates were provided by the Centers for Disease Control and Prevention (CDC), Division of Population Health, Epidemiology and Surveillance Branch. This dataset includes estimates for 40 measures: 12 for health outcomes, 7 for preventive services use, 4 for chronic disease-related health risk behaviors, 7 for disabilities, 3 for health status, and 7 for health-related social needs. These estimates can be used to identify emerging health problems and to help develop and carry out effective, targeted public health prevention activities.  \n")
display(df)
# Check for missing data
m = df.isnull().sum()
print(f"\n Missing data in each column: \n {m}")


'''Question #2'''
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

combine = [] #combined data

#run response api
response = requests.get(f"{base}{endpoint}", params=ptemp)
if response.status_code == 200:
    data = response.json()
    #print(data)
    if 'Data' in data:
        combine.extend(data['Data'])
        #d2 = pd.DataFrame(data['Data'])  
    else:
        print('uh oh')

#display the output and table
if combine:
    d2 = pd.DataFrame(combine)
    print(f"\n AQS contains ambient air sample data collected by state, local, tribal, and federal air pollution control agencies from thousands of monitors around the nation. It also contains meteorological data, descriptive information about each monitoring station (including its geographic location and its operator), and information about the quality of the samples.\n")
    print(f'\n This table displays the daily summary FRM/FEM PM2.5 data for Mobile County Alabama for the years 2019-2021 \n')
    display(d2)
# Check for missing data
m = d2.isnull().sum()
print(f"\n Missing data in each column: \n {m}")



'''Question #3'''

#input info
i_code = ['NY.GDP.MKTP.PP.KD', 'EN.ATM.CO2E.KT']
eu_countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden"
]
#print(wb.economy.coder(eu_countries))
#c = wb.economy.coder(eu_countries)
#grabbing the codes
iso_codes = []
for country in eu_countries:
    lookup = wb.economy.coder(country)
    if lookup:
        iso_codes.append(lookup)

#print(iso_codes)
data = wb.data.DataFrame(i_code, iso_codes, range(1994,2012))
data.to_csv('EU_WBD.csv', index=True) #output a CSV file
print(data) #to also show the display
#df = pd.DataFrame(data)


#End of HW######
'''Previous attempt at using the API that failed'''

# #base_url = 'http://api.worldbank.org/v2'

# def getIndicator(countries, i_id, date_range='1994', per_page=5000, page=1):

#     # countries_str = ';'.join(countries) #so there can be multiple
#     # i_str = ';'.join(i_id)

#     url = (f"{base_url}/country/{countries_str}/indicator/{i_str}"
#           f"?date={date_range}&format=json&per_page={per_page}&page={page}")
#     #url = f"{base_url}/country/all/indicator/SP.POP.TOTL?date=2000:2001"
#     url = 'http://api.worldbank.org/v2/country/chn;bra/indicator/DPANUSSPB?date=2012M01'
#     # ?format=json&per_page={per_page}&page={page}

#     # params = {
#     #     'date': date_range,
#     #     'format': format,
#     #     'per_page': 5000,  # Increase the number of records per page
#     #     'page': 1  # Page number to fetch
#     # }
    

#     response = requests.get(url)
#     response.raise_for_status()
#     if response.status_code == 200:
#         data = response.json()
#         print(f'data {data}')
#         return response
#     else:
#         print("bad response code")

#     # if isinstance(data, list) and len(data) > 1:
#     #     return data[1]  # Data is usually in the second element
#     # else:
#     #     print("Unexpected data format or no data available.")
#     #     return None

# #indicator_code= ['NY.GDP.MKTP.PP.KD', EN.ATM.CO2E.KT]
# #['NY.GDP.MKTP.PP.KD']
#            
# #countries = ['at', 'be', 'bg', 'hr', 'cy', 'cz', 'dk', 'ee', 'fi', 'fr', 'de',
# #     'gr', 'hu', 'ie', 'it', 'lv', 'lt', 'lu', 'mt', 'nl', 'pl', 'pt',
# #     'ro', 'sk', 'si', 'es', 'se']

# info = getIndicator()
# if info:
#     print("hello I am here")
#     df = pd.DataFrame(data= info)
#     #print(df)
#     print(df.head()) #not all the info

