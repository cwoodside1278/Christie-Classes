'''Christie Woodside Homework3 World Bank Sept 10'''

import wbgapi as wb
import pandas as pd
'''info on the indicators'''
# EN.ATM.CO2E.PC – CO2 emissions (metric tons per capita)
# NY.GDP.PCAP.PP.KD – GDP per capita, PPP (constant 2017 international $)
# GB.XPD.RSDV.GD.ZS – Research and development expenditure (% of GDP)
# NE.TRD.GNFS.ZS – Trade (% of GDP)
# EG.FEC.RNEW.ZS – Renewable energy consumption (% of total final energy consumption)
# TX.VAL.TECH.MF.ZS – Medium and high-tech industry (value added % of manufacturing)
# SP.URB.TOTL.IN.ZS – Urban population (% of total population)
# EG.USE.PCAP.KG.OE – Energy use (kg of oil equivalent per capita)

'''The data should be retrieved for all available years and then filtered to include only the years 
from 1990 to 2023. Once the data is filtered, save the dataset as a CSV file for further analysis. 
Be sure to properly label the dataset and ensure that the indicators are clearly identified.'''

i_code = [
    'EN.ATM.CO2E.PC',    # CO2 emissions (metric tons per capita)
    'NY.GDP.PCAP.PP.KD', # GDP per capita, PPP (constant 2017 international $)
    'GB.XPD.RSDV.GD.ZS', # R&D expenditure (% of GDP)
    'NE.TRD.GNFS.ZS',    # Trade (% of GDP)
    'EG.FEC.RNEW.ZS',    # Renewable energy consumption (% of total final energy consumption)
    'TX.VAL.TECH.MF.ZS', # Medium and high-tech industry (value added % of manufacturing)
    'SP.URB.TOTL.IN.ZS', # Urban population (% of total population)
    'EG.USE.PCAP.KG.OE'  # Energy use (kg of oil equivalent per capita)
]

'''Creating the dataframe and CSV file'''
data = wb.data.DataFrame(i_code, time=range(1990, 2024), labels=True)
print(data.head(25))
data.to_csv('wb_1990_2023.csv', index=False)
########################Commenting out so I can use the csv itself

#reading in the new dataframe
df = pd.read_csv('wb_1990_2023.csv')
#print(df.head())

'''Filter the data for observations only for the country "USA".'''
usa_data = df[df['Country'] == 'United States']
print(usa_data.head(10))

'''Filter the data for observations only for the year 2020.'''
#for the United States specifically
year_data = df[(df['Country'] == 'United States')][['Country','Series','YR2020']]
print(f'Year 2020 info for United States\n {year_data.head()}')

#for all the countries
year_data = df[['Country','Series','YR2020']]
print(f'Year 2020 info for all countries\n {year_data}')


'''Filter the data for observations only for the indicator 'EN.ATM.CO2E.PC' (CO2 emissions).'''
#all countries for just co2 indicator
co2_data = df[(df['Series'] == 'CO2 emissions (metric tons per capita)')][['Country','Series','YR2020']]
print(f'C02 indicator for all countries:\n {co2_data}')

#C02 for just United States
co2_data = df[(df['Series'] == 'CO2 emissions (metric tons per capita)')
              & (df['Country'] == 'United States')][['Country','Series','YR2020']]
print(f'C02 indicator for United States:\n {co2_data.head()}\n')


'''Filter the CO2 emissions data for the years between 1994 and 2000.'''
year_col = [f'YR{year}' for year in range(1994, 2001)] 
Y = [col for col in year_col if col in df.columns] #check that it exists
co2_data_year_range = df[(df['Series'] == 'CO2 emissions (metric tons per capita)')][['Country', 'Series'] + Y]
print(f'C02 emissions between 1994 and 2000\n{co2_data_year_range}')


'''Filter the data for the country "USA" and the indicator 'EN.ATM.CO2E.PC' for the years 1994-2000.'''
#I set the values above
co2_data_year_range = df[(df['Series'] == 'CO2 emissions (metric tons per capita)')
                         & (df['Country'] == 'United States')][['Country', 'Series'] + Y]
print(f'C02 emissions between 1994 and 2000 for the United States\n{co2_data_year_range.head()}')



'''Remove any rows with missing data using dropna().'''
clean_df = df.dropna()
print(f'Dataframe with removeds rows that are missing data\n {clean_df.head(10)}')


'''Set a multi-index based on country and date. And use that dataframe ins what follows
  -Filter the data to show only observations for the USA and the indicator CO2 emissions (metric tons per capita).
 -Calculate the average CO2 emissions for USA over the available years.'''

df_l = df.melt(id_vars=['Country', 'Series'], var_name='Year', value_name='Value')
df_l = df_l.dropna() #removing missing values

pivot_df = df_l.pivot_table(index=['Country', 'Year'], columns='Series', values='Value')
print(f'New multi-index dataframe (snippet):\n{pivot_df.head(5)}')


'''_________Filtering the data___________________________________________________________'''
df_l = df.melt(id_vars=['Country', 'Series'], var_name='Year', value_name='Value')
df_l = df_l.dropna() 

dat = df_l[(df_l['Series'] == 'CO2 emissions (metric tons per capita)')
              & (df_l['Country'] == 'United States')][['Country','Series','Year', 'Value']]
#print(dat.head())
co2_data = dat.pivot_table(index=['Country', 'Year'], columns='Series', values='Value', aggfunc='mean')
print(f'C02 indicator for United States\n {co2_data.head(35)}')

'''____Calculate average C02 Emissions_______________________________________________________________'''
m = co2_data['CO2 emissions (metric tons per capita)'].mean()
print(m)
n = dat['Value'].mean()
print(f'Average C02 emissions for USA over the available years{n}')
#co2_data = dat.pivot_table(index=['Country', 'Year'], columns='Series', values='Value')
#print(f'C02 indicator for United States\n {co2_data.head(35)}')
