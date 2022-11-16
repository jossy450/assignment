# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 08:25:27 2022

@author: Eniola
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Importing Dataset on Covid-19 India case time series """

data = pd.read_csv('C:/Users/Eniola/Desktop/UH-courses/ADS1/Week6/covidcases/case_time_series.csv')

""" The dataset has 7 columns. We will be collecting Daily Confirmeed Daily Recovered and Daily Deceased 
in variables as array """
Y = data.iloc[61:,1].values
R = data.iloc[61:,3].values
D = data.iloc[61:,5].values
X = data.iloc[61:,0] #Stores Date


def line_chart(ax, xtick,yticks,tick, title):
    """The function plots line graph """

plt.figure(figsize=(25,8))
""" This creates a canvas for the graph where the first value ‘25’ is the width argument position and ‘8’ is the height argument position of the graph. """
ax = plt.axes()
""" Let’s create an object of the axes of the graph as ‘ax’ so it becomes easier to implement functions """
ax.grid(linewidth=0.4, color='#8f8f8f')

""" set_facecolor’ lets you set the background color of the graph which over here is black. ‘.set_xlabel’ and ‘.set_ylabel’ lets you set the label along both axes whose size and color can be altered """
ax.set_facecolor("black")
ax.set_xlabel('\nDate',size=25,color='#4bb4f2')
ax.set_ylabel('Number of Confirmed Cases\n',
			size=25,color='#4bb4f2')

""" So now we are going to change the ticks of the axes and also annotate the plots. 
 To make the ticks easily readable we change the font color to white and size to 20.
tick_params lets you alter the size and color of the dashes which look act like a bridge between the ticks and graph.
"""
xticks = plt.xticks(rotation='vertical',size='20',color='white')
yticks = plt.yticks(size=20,color='white')
tick = plt.tick_params(size=20,color='white')

""" Annotate lets you annotate on the graph. Over here we have written a code to annotate the plotted points 
by running a for loop which plots at the plotted points. The str(j) holds the Y variable which is 'Daily 
Confirmed'. Any string passed will be plotted. The XY is the coordinates where the string should be plotted. 
And finally the color and size can be defined. Note we have added +100 to j in XY coordinates so that the string 
doesn't overlap with the marker and it is at the distance of 100 units on Y?—?axis """
for i,j in zip(X,Y):
    ax.annotate(str(j),xy=(i,j+100),color='white',size='13')
    ax.annotate('Second Lockdown 15th April',
                xy=(15.2, 860),
                xytext=(19.9,500),
                color='white',
                size='25',
                arrowprops=dict(color='white',
                                linewidth=0.025))

""" To annotate an arrow pointing at a position in graph and its tail holding the string we can define ‘arrowprops’ 
argument along with its tail coordinates defined by ‘xytext’. Note that ‘arrowprops’ alteration can be done using 
a dictionary, 
Finally we define a title by '.title' function and passing string ,its size and color  """
title = plt.title("COVID-19 IN : Daily Confirmed\n",
		size=50,color='#28a9ff')

""" then plot """
ax.plot(X,Y,
		color='#1F77B4',
		marker='o',
		linewidth=4,
		markersize=15,
		markeredgecolor='#035E9B')

line_chart(ax, xticks,yticks,tick, title)




""" Ploting the Pie Chart  """

""" So I have created list slices based on which our Pie Chart will be divided and 
the corresponding activities are it’s valued """
slices = [62, 142, 195]
activities = ['Travel', 'Place Visit', 'Unknown']

cols=['#4C8BE2','#00e061','#fe073a']
exp = [0.2,0.02,0.02]

plt.pie(slices,labels=activities,
		textprops=dict(size=25,color='black'),
		radius=3,
		colors=cols,
		autopct='%2.2f%%',
		explode=exp,
		shadow=True,
		startangle=90)

plt.title('Covid-19 Transmission\n\n\n\n',color='#4fb4f2',size=40)










""" Ploting Bar Chart """
# Load the data to pandas
data = pd.read_csv('C:/Users/Eniola/Desktop/UH-courses/ADS1/Week6/covidcases/district.csv')
data.head()

""" The dataset has 6 columns. We will be collecting data in variables as array """
re=data.iloc[:30,5].values
de=data.iloc[:30,4].values
co=data.iloc[:30,3].values
x=list(data.iloc[:30,0])


plt.figure(figsize=(25,10))
ax=plt.axes()

# Set Face color
ax.set_facecolor('black')
ax.grid(linewidth=0.4, color='#8f8f8f')


plt.xticks(rotation='vertical',
		size='20',
		color='white')#ticks of X

plt.yticks(size='20',color='white')


ax.set_xlabel('\nDistrict',size=25,
			color='#4bb4f2')
ax.set_ylabel('No. of cases\n',size=25,
			color='#4bb4f2')


plt.tick_params(size=20,color='white')


ax.set_title('Maharashtra District wise breakdown\n',
			size=50,color='#28a9ff')

plt.bar(x,co,label='re')
plt.bar(x,re,label='re',color='green')
plt.bar(x,de,label='re',color='red')

for i,j in zip(x,co):
	ax.annotate(str(int(j)),
				xy=(i,j+3),
				color='white',
				size='15')

plt.legend(['Confirmed','Recovered','Deceased'],
		fontsize=20)






