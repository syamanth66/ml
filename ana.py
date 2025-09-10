import pandas as pd
import csv

# Read the CSV file
df = pd.read_csv('data.csv')

# Open the new CSV file for writing
with open('data2.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(['Region/Country/Area', 'Population density and surface area', 'Year', 'Series', 'Value'])
    
    # Iterate through DataFrame rows
    for index, row in df.iterrows():
        region = row['Region/Country/Area']
        population = row['Population density and surface area']
        year = row['Year']
        series = row['Series']
        value = row['Value']
        
        writer.writerow([region, population, year, series, value])
