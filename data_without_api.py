# First install the official client: pip install police-api-client
from police_api import PoliceAPI
import pandas as pd
import matplotlib.pyplot as plt

api = PoliceAPI()  # No API key needed

path_to_csv = "data\\2025-02\\2024-06\\2024-06-city-of-london-street.csv"
# # Get available data months (formatted as YYYY-MM)
# available_dates = api.get_dates()
# latest_month = api.get_latest_date()

# # Get last month's data (assuming current month is latest)
# last_month = "2024-09"  # Replace with your calculated value
# crimes = api.get_crimes_point(
#     lat=51.5074,  # Example: London coordinates
#     lng=-0.1278,
#     date=last_month
# )

# # Print basic crime info
# for crime in crimes[:5]:  # First 5 results
#     print(f"{crime.category}: {crime.location.name}")
#     print(f"Date: {crime.month}, Outcome: {crime.outcomes[0].category if crime.outcomes else 'Pending'}")

# Load the CSV file
data = pd.read_csv(path_to_csv)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Print the column names to see what they look like
print("Column names:", data.columns.tolist())

# Assuming the CSV has 'crime_category' and 'neighborhood' columns
# Adjust the column names based on your CSV structure
crime_counts = data.groupby(['Crime type', 'LSOA name']).size().reset_index(name='counts')

# Print the crimes and their neighborhoods
print(crime_counts)

# Plotting the data
plt.figure(figsize=(10, 6))
for neighborhood in crime_counts['LSOA name'].unique():
    subset = crime_counts[crime_counts['LSOA name'] == neighborhood]
    plt.bar(subset['Crime type'], subset['counts'], label=neighborhood)

plt.xlabel('Crime Category')
plt.ylabel('Number of Crimes')
plt.title('Crimes by Neighborhood')
plt.xticks(rotation=45)
plt.legend(title='Neighborhood')
plt.tight_layout()
plt.show()
