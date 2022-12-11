
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def read_files(file_name):
    """A function that reads world bank data and
       generates teo dataframes: df1 and a transposed
       version of df1.

    Args:
        file_name (csv file): Name/ path to the file to be read.

    Returns:
        [tuple]: [a tuple of two pandas datafrmae]
    """    
    df1 = pd.read_csv(file_name, 
                      skiprows =4)
    
    #transpose df1, reset index and grab all but the first row
    df2 = df1.T.reset_index().iloc[:,1:] 
    new_header = df2.iloc[0]   #grab the first row for the header
    df2 = df2[1:]               #take the data rows alone
    df2.columns = new_header   #set the header row as the dataframe header
     
    #return the dataframes    
    return df1, df2

def line_plotter(df, x_axis, y_axis, legend, title):
    """This functions plots a line chart

    Args:
        df (ataframe object): _description_
        x_axis (string): Name of the column to be on the x-axis of the line chart.
        y_axis (string): Name of the column on the Y-axis of the line chart.
        legend (string): Name of column whose values serves as legend in the graph.
        title (string): Chart title
    """
    fig, ax = plt.subplots(figsize=(16, 8), dpi=200)

    for value in df[legend].unique():
        
        #plot values
        df[df[legend] == value].plot(x=x_axis, 
                                     y=y_axis, 
                                     ax=ax, 
                                     label=value, 
                                     linestyle = '--',  
                                     markersize = 2)
    
    #Legend location
    plt.legend(loc='upper right', 
               fontsize = 12)
    
    #Assign xticks
    plt.xticks(df[x_axis].unique())
    
    #Assign label to y-axis
    plt.ylabel(y_axis+' (% of total)')
    
    #plot title
    plt.title(title, fontsize =20)
    
    plt.show()

def bar_plotter(data, x_axis, y_axis, color, title, _type):
    """A function that plots a bar chart.

    Args:
        data (dataframe): Dataframe that holds the plot values
        x_axis (string): Name of the column to be on the x-axis of the Bar chart.
        y_axis (string): Name of the column on the Y-axis of the Bar chart.
        color (String): Chart colour.
        title (String): Chart title.
        _type (String): Bar type: barh or bar
    """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)

    data.plot(kind=_type, 
              x= x_axis, 
              y= y_axis, 
              ax=ax, 
              color = color)
    
    plt.xticks(rotation=0)
    plt.title(title, font = 'Segoe UI', fontsize = 20)
    
    if _type == 'barh':
        plt.xlabel(y_axis, fontsize = 15)
    else:
        plt.ylabel(y_axis, fontsize = 15)
        
    ax.get_legend().remove()
    
    plt.show()


def heatmap_plotter(dataframe, title):
    """A functions that plots correlation heatmap
       and returns p-value

    Args:
        dataframe (dataframe object): Dataframe of values to be correlated.
        title (string): Chart title.

    Returns:
        dataframe object: A datafrmae of p-values
    """    
    #select useful columns
    data = dataframe.iloc[:,3:]
    
    df_corr = pd.DataFrame() # Correlation matrix
    df_p = pd.DataFrame()  # p-values matrix
    
    for x in data.columns:
        for y in data.columns:
            #calculate pearson correlation and p-values
            corr = stats.pearsonr(data[x], data[y])
            df_corr.loc[x,y] = corr[0]
            df_p.loc[x,y] = corr[1]
    
    #Generate heatmap for the correlation values
    ax = plt.subplots(figsize=(10, 6), dpi=200)
    sns.heatmap(df_corr, 
                cmap="YlGnBu", 
                annot=True)

    #chart title
    plt.title(title, fontsize = 20)

    #Display plot
    plt.show()
    
    #return p-values
    return df_p

def data_slicer(dataframe,column_name, slicer):
    """A function that slices data based on column and slicer value.

    Args:
        dataframe (dataframe): dataframe to be sliced
        column_name (string): column that contains the slicing condition
        slicer (string): Slicing condition, usually row value

    Returns:
        pd.dataframe: A dataframe 
    """    
    data = dataframe[dataframe[column_name]==slicer]
    
    #return sliced data
    return data
    
#read worldbank data with the defined function
tables = read_files('C:/Users/Eniola/Downloads/API_19_DS2_en_csv_v2_4700503.csv')


#Get the first dataframe from the returned tuple
main_data = tables[0]

main_data.head()

#check for null values
main_data.isnull().sum()


#remove duplicates
main_data.drop_duplicates(inplace = True)


#select the first 66 columns
main_data = main_data.iloc[:, 0:66 ] 

#convert data from long format to wide format
long_main_data = main_data.melt(id_vars = ['Country Name',
                                           'Country Code',
                                           'Indicator Name',
                                           'Indicator Code'],
                                var_name = 'year',
                                value_name = 'value')

#check 20 samples from the long data
long_main_data.sample(20)

#Get population data with data_slicer function
population_data = data_slicer(long_main_data,
                              'Indicator Name', 
                              'Urban population (% of total population)')

population_data.head()

#selected needed columns from population data
population = population_data[['Country Name', 'Country Code', 'year', 'value']].copy()


#rename 'value' to 'urban_population'
population.rename(columns = {'value':'urban_population'}, inplace = True)

#get 'Agriculture, forestry, and fishing, value added (% of GDP)' data
agric_value_data = data_slicer(long_main_data, 
                               'Indicator Name',
                               'Agriculture, forestry, and fishing, value added (% of GDP)')


agric_value_data.head()

#select needed columns
agric_value = agric_value_data[['Country Name', 'Country Code', 'year', 'value']].copy()

#rename 'value' to 'agric_value'
agric_value.rename(columns = {'value':'agric_value'}, inplace = True)

agric_value.head()


agricultural_land_data = data_slicer(long_main_data,
                                     'Indicator Name',
                                     'Agricultural land (% of land area)')

agricultural_land_data.head()


#select needed columns from agricultural_land_data
agricultural_land = agricultural_land_data[['Country Name', 'Country Code', 'year', 'value']].copy()


#Rename 'value' to 'agricultural_land'
agricultural_land.rename(columns = {'value':'agricultural_land'}, inplace = True)

agricultural_land.sample(10)

#Join population and agric_value
combined_data = pd.merge(population, agric_value,  
                        how='left', 
                        left_on=['Country Name', 'Country Code', 'year'], 
                        right_on=['Country Name', 'Country Code', 'year'])

#Join combined_data and agricultural_land
combined_data = pd.merge(combined_data, agricultural_land,
                        how='left', 
                        left_on=['Country Name', 'Country Code', 'year'], 
                        right_on=['Country Name', 'Country Code', 'year'])


#change 'year' column to int datatype
combined_data['year'] =combined_data['year'].astype(int)


#select rows where year is greater than 1995
combined_data = combined_data[combined_data['year']>1995].copy()

combined_data = combined_data.sort_values(['year', 'urban_population'], ascending = False)


#Countries to focus on 
focused_countries = ['China', 'India','United States', 
                     'Indonesia','Pakistan', 'Brazil', 
                     'Nigeria', 'Bangladesh','Russian Federation',
                     'Mexico']

#select data for focused countries
focus_data = combined_data[combined_data['Country Name'].isin(focused_countries)]


#Remove 2021 data from focus_data as majority are null
focus_data = focus_data[focus_data['year']<2021].copy()

line_plotter(focus_data, 
             'year', 
             'urban_population',
             'Country Name','Trends in Urban Population (1996-2020)')

line_plotter(focus_data, 
             'year', 
             'agric_value',
             'Country Name','Changes Agricultural Value Added to Economy (1996-2020)')

line_plotter(focus_data, 
             'year', 
             'agricultural_land',
             'Country Name','Changes in Agricultural land (1996-2020)')

changes_data = main_data[['Country Name','Indicator Name', '1996', '2020']].copy()

urban_changes = changes_data[changes_data['Indicator Name']=='Urban population (% of total population)'].copy()

#calculate percentage change between 1996 and 2020
urban_changes['percent_change'] = urban_changes['2020']-urban_changes['1996']

#Get urban_changes data for countries in focused_countries
urban_changes = urban_changes[urban_changes['Country Name'].isin(focused_countries)]

#bar plot for urban_changes
bar_plotter(urban_changes.sort_values('percent_change', ascending=False), 
            'Country Name', 
            'percent_change',
            'blue', 
            'Percentage Changes in Urban Population (1996- 2020)',
            'bar')

#plot heatmap for focus_data
heatmap_plotter(focus_data.dropna(), 'General Correlation for Selected Countries')
#slice out China data only
china_data = data_slicer(focus_data, 'Country Name', 'China')

#plot heatmap for china data
heatmap_plotter(china_data, 'China Heatmap')


#Heatmap and p-values for United States data
heatmap_plotter(data_slicer(focus_data.dropna(), 
                            'Country Name', 
                            'United States'),
                'United States Heatmap')

#Heatmap and p-values for Indonesia
heatmap_plotter(data_slicer(focus_data, 
                            'Country Name', 
                            'Indonesia'),
                'Indonesia Heatmap')

#Heatmap and p-values for Russia
heatmap_plotter(data_slicer(focus_data, 
                            'Country Name', 
                            'Russian Federation'),
                'Russian Federation Heatmap')





