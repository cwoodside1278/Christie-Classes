# %%
# Import required libraries
import wbgapi as wb
import pandas as pd

# %%

print("""1. Please download the data from the World Bank’s data API for all countries, focusing on the following key macroeconomic and environmental indicators:

CO2 emissions (metric tons per capita)
GDP per capita (PPP, constant 2017 international $)
Research and development expenditures (% of GDP)
Trade (% of GDP)
Renewable energy consumption (% of total final energy consumption)
Medium and high-tech industry value-added (% of manufacturing)
Urban population (% of total population)
Energy use (kg of oil equivalent per capita)
The data should be retrieved for all available years and then filtered to include only the years from 1990 to 2023. Once the data is filtered, save the dataset as a CSV file for further analysis. Be sure to properly label the dataset and ensure that the indicators are clearly identified.

""")


# Define the indicators we want to download
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

# Download data from World Bank API
raw_data = wb.data.DataFrame(list(indicator_dict.keys()))

# Clean up the column names and rename indicators
cleaned_data = raw_data.rename(columns=lambda x: x.lstrip("YR") if x.startswith("YR") else x)
cleaned_data = cleaned_data.rename(index=indicator_dict)

# Set up multi-index and convert year columns to numeric
cleaned_data.index.set_names(["Country", "Indicator"], inplace=True)
cleaned_data.columns = pd.to_numeric(cleaned_data.columns)
cleaned_data.columns.name = "Year"

# Filter data for years 1990-2023
filtered_data = cleaned_data.loc[:, 1990:2023]

# %%

# Save the dataset as a CSV file
filtered_data.to_csv("wb_data.csv")

print("Data has been successfully downloaded, processed, and saved to 'world_bank_data_1990_2023.csv'")

# Optional: Display some basic information about the dataset

print("\nDataset Information:")
print(f"Number of countries: {filtered_data.index.get_level_values('Country').nunique()}")
print(f"Number of indicators: {len(indicator_dict)}")
print(f"Year range: {filtered_data.columns.min()} - {filtered_data.columns.max()}")
print("\nFirst few rows of the dataset:")
print(filtered_data.head())


# %%

print("""a. Filter data for the USA only""")
df_usa = filtered_data.loc["USA"]
print(df_usa.head())

# %%

print("""b. Filter data for the year 2020 only""")
df_2020 = filtered_data.loc[:, 2020]
print(df_2020.head())

# %%

print("""c. Filter data for CO2 emissions only""")
df_co2 = filtered_data.loc[(slice(None), "CO2 emissions"), :]
print(df_co2.head())

# %%

print("""d.Filter CO2 emissions data for 1994-2000""")
df_co2_1994_2000 = filtered_data.loc[(slice(None), "CO2 emissions"), 1994:2000].droplevel("Indicator")
print(df_co2_1994_2000.head())

# %%

print("""e.Filter data for USA and CO2 emissions in 1994-2000""")
df_usa_co2_1994_2000 = filtered_data.loc[("USA", "CO2 emissions"), 1994:2000]
print(df_usa_co2_1994_2000.head())

# %%

print("""f.Remove rows with missing data""")
filtered_data.dropna()
print(filtered_data.head())

# %%

print("""g.Set multi-index and calculate average CO2 emissions for USA""")
filtered_data.unstack("Indicator").stack("Year")
df_usa_co2_mean = filtered_data.loc[("USA", "CO2 emissions")].mean()
print(df_usa_co2_mean)

# %%
