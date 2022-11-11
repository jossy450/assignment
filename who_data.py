# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:58:09 2022

@author: Eniola
"""
# Calling libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#Use pandas to read the file
df = pd.read_csv("C:/Users/Eniola/Downloads/vaccination_data.csv")

# This will show the columns name and other related information
df.info()

#To avoid errors, converting dates to pandas datetime
df['date_updated'] = pd.to_datetime(df['date_updated'])
df['first_vaccine_date'] = pd.to_datetime(df['first_vaccine_date'])

df['month'] = pd.DatetimeIndex(df['date_updated']).month
df['day'] = pd.DatetimeIndex(df['date_updated']).day
df['weekday'] = pd.DatetimeIndex(df['date_updated']).weekday
df['weekofyear'] = pd.DatetimeIndex(df['date_updated']).weekofyear


df['month'] = pd.DatetimeIndex(df['first_vaccine_date']).month
df['day'] = pd.DatetimeIndex(df['first_vaccine_date']).day
df['weekday'] = pd.DatetimeIndex(df['first_vaccine_date']).weekday
df['weekofyear'] = pd.DatetimeIndex(df['first_vaccine_date']).weekofyear


# Select neccesary columns for plotting after cleansing
data = df[['who_region','continents', 'total_vaccinations', 'persons_fully_vaccinated', 'persons_booster_add_dose']]

# Summarise the selected data
sumit = [df['total_vaccinations'].sum(), df['persons_fully_vaccinated'].sum(),
           df['persons_booster_add_dose'].sum()]

# Add Summarized data to diplay in pandas dataframe
sum_sumit = pd.DataFrame(sumit,['total_vaccinations','persons_fully_vaccinated','persons_booster_add_dose'],columns=['Sum'])

sum_sumit

# Define a function bar_chart
def bar_chart(_axis, list,title):
    plt.figure(figsize=(12,8))
    plt.bar(j_axis,list)
    plt.title(title, fontsize= 8)
    plt.show()
    return

# Using the function to Plot
j_axis = sum_sumit.index
sum_list = sum_sumit['Sum']
brand = 'This Bar Chart depicts Data from WHO on total vaccinations administered worldwide'
plt.legend(loc="upper right")
bar_chart(j_axis,sum_list,brand)

worldwide = data.groupby('continents')[['total_vaccinations','persons_fully_vaccinated','persons_booster_add_dose']].sum()
worldwide


# Define a function line_plot
def line_plot(x_axis,my_list,xticks,label,title):
    plt.figure(figsize=(8,5))
    for i in range(len(my_list)):
        plt.plot(x_axis,my_list[i],label=label[i])
    plt.xticks(x_axis,xticks)
    plt.title(title,fontsize=6)
    plt.show()
    
# Using the function to plot
    x_axis = worldwide.index
    my_list = [worldwide['total_vaccinations'],worldwide['persons_fully_vaccinated'],worldwide['persons_booster_add_dose']]
    xticks = ['Africa','America','Asian','Europe']
    label = ['Total Vaccinations Worldwide', 'Persons Fully Vaccinated', 'Booster Dose Received Worldwide']
    title = 'This Line Plot depicts Data from WHO on Total Number of vaccinations administered worldwide'
    line_plot(x_axis,my_list,xticks,label,title)


#Using the function to plot

who_region = data.groupby('who_region')[['total_vaccinations','persons_fully_vaccinated','persons_booster_add_dose']].sum()
who_region

def subplot_pie_chart(x_axis,label,title):
    plt.figure(figsize=(15,10))
    for i in range(len(x_axis)):
        plt.subplot(2,2,i+1).set_title(title[i])
        plt.pie(x_axis[i],labels=label)
    plt.show()
    
    x_axis = [who_region['total_vaccinations'], who_region['persons_fully_vaccinated'], who_region['persons_booster_add_dose']]
    label = who_region.index
    title = ['Total Vaccinations Worldwide','Persons Fully Vaccinated','Booster Dose Received Worldwide']

    subplot_pie_chart(x_axis,label,title)







    




