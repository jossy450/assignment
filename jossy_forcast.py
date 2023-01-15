import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# inline magic command to show matplotlib plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

def load_and_clean(data1, data2):
    """
    Load and clean data from the two input files, data1 and data2.
    The function reads the data with pandas, drops unnecessary columns, melts the data to reshape it,
    and concatenates the two dataframes.
    It then pivots the dataframe to make the Indicator Name column the columns of the pivot table and reset the index.
    """
    # Read data1 and drop unnecessary column
    df1 = pd.read_csv(data1, skiprows = 4)
    df1 = df1.drop('Unnamed: 66', axis =1)
    # Melt data1 and reshape it
    df1 = df1.melt(id_vars = ['Country Name',
                                           'Country Code',
                                           'Indicator Name',
                                           'Indicator Code'],
                                var_name = 'year',
                                value_name = 'value')
    
    # Read data2 and drop unnecessary column
    df2 = pd.read_csv(data2, skiprows = 4)
    df2 = df2.drop('Unnamed: 66', axis =1)
    # Melt data2 and reshape it
    df2 = df2.melt(id_vars = ['Country Name',
                                           'Country Code',
                                           'Indicator Name',
                                           'Indicator Code'],
                                var_name = 'year',
                                value_name = 'value')
    
    # Concatenate the two dataframes
    dataframe = pd.concat([df1, df2])
    dataframe = dataframe[['Country Name', 'Indicator Name', 'year', 'value']].copy()
    
    # Pivot the dataframe to make the Indicator Name column the columns of the pivot table and reset the index
    dataframe = dataframe.pivot(index=['Country Name', 'year'],
                                columns='Indicator Name', 
                                values='value').reset_index()
    
    return dataframe

# Load and clean data
df = load_and_clean('API_SP.DYN.LE00.IN_DS2_en_csv_v2_4770434.csv', 
                    'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4770417.csv')

# Display the first 5 rows of the dataframe
df.head()

# Display the shape of the dataframe
df.shape

# Display the number of missing values in the dataframe
df.isnull().sum()

# Convert year column to int
df['year'] = df['year'].astype(int)

# List of non-country entities
non_country = ['Africa Eastern and Southern','Arab World','Caribbean small states','Central African Republic', 'Central Europe and the Baltics',
'Early-demographic dividend', 'East Asia & Pacific',
'East Asia & Pacific (IDA & IBRD countries)',
'East Asia & Pacific (excluding high income)','Europe & Central Asia',
'Europe & Central Asia (IDA & IBRD countries)',
'Europe & Central Asia (excluding high income)', 'European Union',
'Fragile and conflict affected situations','French Polynesia','Heavily indebted poor countries (HIPC)',
'High income', 'IBRD only',
'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total','Late-demographic dividend',
'Latin America & Caribbean',
'Latin America & Caribbean (excluding high income)',
'Latin America & the Caribbean (IDA & IBRD countries)',
'Least developed countries: UN classification', 'Low & middle income', 'Low income', 'Lower middle income',
'Middle East & North Africa',
'Middle East & North Africa (IDA & IBRD countries)',
'Middle East & North Africa (excluding high income)',
'Middle income', 'Not classified',
'OECD members', 'Other small states',
'Pacific island small states','Post-demographic dividend',
'Pre-demographic dividend','Small states','South Asia (IDA & IBRD)','Sub-Saharan Africa',
'Sub-Saharan Africa (IDA & IBRD countries)',
'Sub-Saharan Africa (excluding high income)','Upper middle income', 'West Bank and Gaza',
'World','Africa Western and Central'
]

#Filter out rows corresponding to non-country entities
df = df[~df['Country Name'].isin(non_country)]

# Filter data to only include the latest available data (in this case, 2020)
latest_data = df[df['year']==2020]

# Check for missing values and drop them
print(latest_data.isnull().sum())
latest_data = latest_data.dropna()

# Prepare data for clustering by dropping unnecessary columns
cluster_data = latest_data.drop(['Country Name', 'year'], axis = 1)

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(cluster_data)

# Use the elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values to determine the optimal number of clusters
import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Initialize the model with the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the model to the data
kmeans.fit(data_scaled)

# Use the silhouette score to evaluate the quality of the clusters
from sklearn.metrics import silhouette_score
print(f'Silhouette Score: {silhouette_score(data_scaled, kmeans.labels_)}')

# Predict the clusters
y_pred = kmeans.fit_predict(cluster_data)

# Add the cluster predictions to the original data
latest_data['cluster'] = y_pred

# Visualize the clusters
import seaborn as sns
plt.figure(figsize=(12,6))
sns.set_palette("pastel")  
sns.scatterplot(x= latest_data['GDP per capita (current US$)'],
                y= latest_data['Life expectancy at birth, total (years)'], 
                hue= latest_data['cluster'], 
                palette='bright')

plt.title('Country Clusters Based on GDP per Capita and Life Expectancy (2020)', fontsize = 18)

# Repeat the process for the data from 1995
old_data = df[df['year']==1995]

# Check for missing values and drop them
print(old_data.isnull().sum())
old_data = old_data.dropna()

# Prepare data for clustering by dropping unnecessary columns
cluster_df = old_data.drop(['Country Name', 'year'], axis = 1)

#Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(cluster_df)

#Use the elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

#Plot the WCSS values to determine the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Initialize the model with the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

#Fit the model to the data
kmeans.fit(data_scaled)

#Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {silhouette_score(data_scaled, kmeans.labels_)}')

#Predict the clusters
y_pred = kmeans.fit_predict(cluster_df)

#Add the cluster predictions to the original data
old_data['cluster'] = y_pred

#Visualize the clusters
plt.figure(figsize=(12,6))
sns.scatterplot(x= old_data['GDP per capita (current US$)'],
y= old_data['Life expectancy at birth, total (years)'],
hue= old_data['cluster'],
palette='bright')

plt.title('Country Clusters Based on GDP per Capita and Life Expectancy (1995)', fontsize = 18)

for i in [0,1,2]:
    # Get the unique country names for each cluster in 2020 and 1995 data
    f = latest_data[latest_data['cluster']==i]['Country Name'].unique()
    g = old_data[old_data['cluster']==i]['Country Name'].unique()
    print(f'Countries in cluster {i} in 2020:')
    print(f)
    print('*****')
    print(f'Countries in cluster {i} in 1995:')
    print(g)
    print("")

# Plot life expectancy data for Japan
japan = df[df['Country Name'] == 'Japan']
japan = japan.dropna()
plt.figure(figsize=(12,6))
japan.plot( 'year', "Life expectancy at birth, total (years)", figsize =(12,8))
plt.title('Life Expectancy in Japan')
plt.show()

# Define functions for model fitting
def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

# Error range calculation function
def err_ranges(x, func, param, sigma):
    """
    This function calculates the upper and lower limits of function, parameters and
    sigmas for a single value or array x. The function values are calculated for 
    all combinations of +/- sigma and the minimum and maximum are determined.
    This can be used for all number of parameters and sigmas >=1.
    """
    import itertools as iter
    lower = func(x, *param)
    upper = lower
    uplow = []   
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper

# Function to fit and forecast life expectancy data
def fit_and_forecast(function, data, country):
      # Fit the model to the data
    param, covar = opt.curve_fit(logistic, data["year"], data["Life expectancy at birth, total (years)"],
                                 p0=(3e2, 0.03, 2000.0))
    # Plot the fit
    plt.figure(figsize=(12,6))
    data["fit"] = function(data["year"], *param)
    data.plot("year", ["Life expectancy at birth, total (years)", "fit"], figsize =(12,8))
    plt.title( country + " Life Expectancy Model Fit", fontsize = 16)
    plt.show()

    # Generate forecast
    year = np.arange(1960, 2041)
    forecast = logistic(year, *param)

    # Calculate error ranges
    sigma = np.sqrt(np.diag(covar))
    low, up = err_ranges(year, logistic, param, sigma)

    # Plot forecast with error ranges
    data.plot("year", ["Life expectancy at birth, total (years)"], figsize =(12,8))
    plt.plot(year, forecast, label="forecast")
    plt.title(country + " Life Expectancy Forcast", fontsize = 16)
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.legend()
    plt.show()
    
#Fit and forecast life expectancy data for Japan
fit_and_forecast(logistic, japan, "Japan")

# Create a dataframe for Luxembourg
luxembourg = df[df['Country Name']=='Luxembourg'].dropna()

# Plot Life Expectancy vs Year
plt.figure(figsize=(12,6))
luxembourg.plot( 'year', "Life expectancy at birth, total (years)", figsize =(12,8) )
plt.title('Luxembourg GDP Per Capita VS Life Expectancy')
plt.show()

# Fit the exponential function to the data
param, covar = opt.curve_fit(exponential, luxembourg["year"], luxembourg["Life expectancy at birth, total (years)"],
                             p0=(1, 0))

# Add fit column to the dataframe
luxembourg["fit"] = exponential(luxembourg["year"], *param)

# Plot the data and the fit
luxembourg.plot("year", ["Life expectancy at birth, total (years)", "fit"], figsize =(12,8))

# Generate forecast
year = np.arange(1960, 2041)
forecast = exponential(year, *param)

# Calculate error ranges
sigma = np.sqrt(np.diag(covar))
low, up = err_ranges(year, exponential, param, sigma)
    
# Plot forecast with error ranges
luxembourg.plot("year", ["Life expectancy at birth, total (years)"], figsize =(12,8))
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Life Expectancy")
plt.title("Luxembourg Life Expectancy Forcast and Confidence Interval", fontsize = 16)
plt.legend()
plt.show()

# Calculate error ranges for year 2070
print(err_ranges(2070, exponential, param, sigma))

# Create a dataframe for Canada
canada =df[df['Country Name']=='Canada'].dropna()

# Plot Life Expectancy vs Year
plt.figure(figsize=(12,6))
canada.plot( 'year', "Life expectancy at birth, total (years)" , figsize =(12,8))
plt.title('Canada GDP Per Capita VS Life Expectancy')
plt.show()

# Fit the logistic function to the data
param, covar = opt.curve_fit(logistic, canada["year"], canada["Life expectancy at birth, total (years)"],
                              p0=(2500, 0.03, 1990.0))

# Plot the data and the fit
plt.figure(figsize=(12,6))
canada["fit"] = logistic(canada["year"], *param)
canada.plot("year", ["Life expectancy at birth, total (years)", "fit"], figsize =(12,8))
plt.title(" Life Expectancy Model Fit", fontsize = 16)
plt.show()

# Generate forecast
year = np.arange(1960, 2030)
forecast = logistic(year, *param)

# Calculate error ranges
sigma = np.sqrt(np.diag(covar))
low, up = err_ranges(year, logistic, param, sigma)

# Plot forecast with error ranges
canada.plot("year", ["Life expectancy at birth, total (years)"], figsize =(12,8))
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Life Expectancy")
plt.title("Canada Life Expectancy Forcast and Confidence Interval", fontsize = 16)
plt.legend()
plt.show()

# Create a dataframe for Sri Lanka
sri_lanka = df[df['Country Name']== 'Sri Lanka']
sri_lanka = sri_lanka.dropna()
               
# Plot Life Expectancy vs Year
plt.figure(figsize=(12,6))
sri_lanka.plot( 'year', "Life expectancy at birth, total (years)", figsize =(12,8) )
plt.title('Sri Lanka GDP Per Capita VS Life Expectancy', fontsize = 16)
plt.show()

# Fit the exponential function to the data
param, covar = opt.curve_fit(exponential, sri_lanka["year"], sri_lanka["Life expectancy at birth, total (years)"],
                             p0=(1, 0))

# Plot the data and the fit
sri_lanka["fit"] = exponential(sri_lanka["year"], *param)
sri_lanka.plot("year", ["Life expectancy at birth, total (years)", "fit"], figsize =(12,8))
plt.title(" Sri Lanka Life Expectancy Model Fit", fontsize = 16)
plt.show()

# Generate forecast
year = np.arange(1960, 2041)
forecast = exponential(year, *param)

# Calculate error ranges
sigma = np.sqrt(np.diag(covar))
low, up = err_ranges(year, exponential, param, sigma)

# Plot forecast with error ranges
sri_lanka.plot("year", ["Life expectancy at birth, total (years)"], figsize =(12,8))
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Life Expectancy")
plt.title("Sri Lanka Life Expectancy Forcast and Confidence Interval", fontsize = 16)
plt.legend()
plt.show()


               



