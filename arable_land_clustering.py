from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

def read_arable_land(filename):
    """
      The purpose of this function is to read a Worldbank format file containing data on arable land for various countries over time, and return a cleaned pandas DataFrame with the relevant data.
    
      Input:
    
      filename: a string containing the name or path of the Worldbank format file to be read.
    
      Output:
    
      df: a pandas DataFrame containing the cleaned arable land data for each country over time. The columns of the DataFrame represent the years, and the rows represent the countries. The values in the DataFrame are the percentage of arable land for each country in each year.
     """
   
    # Read the Worldbank format file using pandas
    df = pd.read_csv(filename, skiprows=4)

    df = df.fillna(0)
    df=df.drop([ 'Country Code', 'Indicator Code'], axis=1)

    # df_countries = df_years.transpose()
    return df

df_arabale = read_arable_land('Arable land/API_AG.LND.ARBL.ZS_DS2_en_csv_v2_5362201.csv')

df_arabale_temp=df_arabale

df_arabale_countries=df_arabale.transpose()




# Get the last 10 years of data
data_1 = df_arabale.loc[:, "2010":"2020"]
data_2 = df_arabale.loc[:, "2000":"2010"]
data_3 = df_arabale.loc[:, "1990":"2000"]
data_4 = df_arabale.loc[:, "1980":"1990"]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()

transformed_data_1 = scaler.fit_transform(data_1)
transformed_data_2 = scaler.fit_transform(data_2)
transformed_data_3 = scaler.fit_transform(data_3)
transformed_data_4 = scaler.fit_transform(data_3)

# Define the number of clusters
n_clusters = 3

# Apply KMeans clustering algorithm
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_1 = kmeans.fit_predict(transformed_data_1)
labels_2 = kmeans.fit_predict(transformed_data_2)
labels_3 = kmeans.fit_predict(transformed_data_3)
labels_4 = kmeans.fit_predict(transformed_data_4)

# Add the cluster labels to the original dataframe
df_arabale["2010-2020"] = labels_1
df_arabale["2000-2010"] = labels_2
df_arabale["1990-2000"] = labels_3
df_arabale["1980-1990"] = labels_4
# Print the cluster assignments
print(df_arabale[["Country Name", "2010-2020","2000-2010","1990-2000", "1980-1990"]])



# Get the indices of each cluster
C1 = df_arabale[df_arabale["2010-2020"] == 0].index
C2 = df_arabale[df_arabale["2010-2020"] == 1].index
C3 = df_arabale[df_arabale["2010-2020"] == 2].index

# Print the names of countries in each cluster
print("Cluster 1 countries:")
print(df_arabale.loc[C1, "Country Name"].to_string(index=False))
print()
print("Cluster 2 countries:")
print(df_arabale.loc[C2, "Country Name"].to_string(index=False))
print()
print("Cluster 3 countries:")
print(df_arabale.loc[C3, "Country Name"].to_string(index=False))
print()

# Count the number of countries in each cluster for 2010 to 2020
cluster_1=[]
cluster_2=[]
cluster_3=[]
list_cls=['2010-2020','2000-2010','1990-2000','1980-1990']
for cls in list_cls:
  print("clusters in "+str(cls))
  count = df_arabale[cls].value_counts()
  print("Cluster 1 has {} countries".format(count[0]))
  cluster_1.append(count[0])
  print("Cluster 2 has {} countries".format(count[1]))
  cluster_2.append(count[1])
  print("Cluster 3 has {} countries".format(count[2]))
  cluster_3.append(count[2])

def print_yearwise_change(years,cluster_1,cluster_2,cluster_3):
    """
    This function creates a bar plot that shows the distribution of countries across the three clusters
    for different year ranges.

    Parameters:
    years (list of str): A list of strings representing the year ranges for which the distribution is to be plotted.
    cluster_1 (list of int): A list of integers representing the number of countries in cluster 1 for each year range.
    cluster_2 (list of int): A list of integers representing the number of countries in cluster 2 for each year range.
    cluster_3 (list of int): A list of integers representing the number of countries in cluster 3 for each year range.

    Returns:
    None
    """
    # Define the x-axis positions
    x_pos = [i for i, _ in enumerate(years)]

    # Create the bar plot for each cluster
    plt.bar(x_pos, cluster_1, color='blue', edgecolor='black', width=0.3)
    plt.bar([i + 0.3 for i in x_pos], cluster_2, color='orange', edgecolor='black', width=0.3)
    plt.bar([i + 0.6 for i in x_pos], cluster_3, color='green', edgecolor='black', width=0.3)

    # Add axis labels and title
    plt.xlabel("Year Range")
    plt.ylabel("Number of Countries")
    plt.title("Cluster Distribution")

    # Add tick labels
    plt.xticks([i + 0.3 for i in x_pos], years)

    # Add legend
    plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'], loc='upper right')

    # Show the plot
    plt.show()



# Define the data
years = ["2010-2020", "2000-2010", "1990-2000", "1980-1990"]
print_yearwise_change(years,cluster_1,cluster_2,cluster_3)



# Create a figure with 4 subplots, one for each time interval
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), subplot_kw={'projection': '3d'})

# Loop over each time interval
for i, time_interval in enumerate([transformed_data_1, transformed_data_2, transformed_data_3, transformed_data_4]):

    # Loop over each cluster
    for cluster in range(3):
        # Filter the data for the current cluster
        cluster_data = time_interval[labels_1 == cluster]
        cluster_name = 'Cluster {}'.format(cluster + 1)

        # Create a scatter plot with color-coded points based on cluster assignments
        axs[i//2][i%2].scatter(cluster_data[:, -3], cluster_data[:, -2], cluster_data[:, -1], cmap='viridis', s=100, label=cluster_name)

    # Add axis labels and legend for each subplot
    axs[i//2][i%2].set_xlabel('', fontsize=12)
    axs[i//2][i%2].set_ylabel('', fontsize=12)
    axs[i//2][i%2].set_zlabel('', fontsize=12)
    axs[i//2][i%2].legend(fontsize=10)

    # Add a title for each subplot
    axs[i//2][i%2].set_title('{}'.format(['2010-2020', '2000-2010', '1990-2000', '1980-1990'][i]), fontsize=14)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

df_arabale = read_arable_land('Arable land/API_AG.LND.ARBL.ZS_DS2_en_csv_v2_5362201.csv')
df_arabale_temp=df_arabale
Selected_country='United Kingdom'
country_data = df_arabale_temp.loc[df_arabale_temp['Country Name'] == Selected_country].iloc[:, 2:-2].T
years = np.array(range(1960, 2021))

country_data

years = pd.DataFrame(years)

years=years.squeeze()

country_data=country_data.squeeze()




# Extract the relevant data columns
year = years
attribute = country_data
# Linear model
def linear_model(x, a, b):
    return a * x + b

# Fit the model to the data
popt, pcov = curve_fit(linear_model, year, attribute)

# Use the model to make predictions for the next 20 years
future_years = np.arange(year.min(), year.max() + 20)
future_attribute = linear_model(future_years, *popt)
print(future_years)
print(future_attribute)
# Estimate the confidence range using the attached function err_ranges
def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    y = linear_model(x, *popt)
    low = y - perr[0] * np.exp(popt[1] * (x - year.min()))
    high = y + perr[0] * np.exp(popt[1] * (x - year.min()))
    return low, high

future_low, future_high = err_ranges(popt, pcov, future_years)

# Plot the data, model, and confidence range
plt.plot(year, attribute, 'bo', label='Data')
plt.plot(future_years, future_attribute, 'r-', label='Model')
plt.fill_between(future_years, future_low, future_high, alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Arable Land')
plt.title('Exponential Model of '+str(Selected_country))
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Define the x and y data
x = future_years
y = future_attribute

# Plot the data with a dashed red line
plt.plot(x, y, 'r--', linewidth=2)

# Add grid lines
plt.grid(True)

# Customize the axis labels and title
plt.xlabel('Year', fontsize=12, color='gray')
plt.ylabel('Arable Land', fontsize=12, color='gray')
plt.title('Linear model of '+str(Selected_country), fontsize=14, fontweight='bold', color='navy')

# Add a legend
plt.legend([''], loc='upper left')

# Show the plot
plt.show()