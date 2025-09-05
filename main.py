import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_routes = 5
num_days = 30
dates = pd.to_datetime(['2024-03-01'])+ pd.to_timedelta(np.arange(num_days), unit='D')
data = {
    'Date': np.repeat(dates, num_routes),
    'Route': np.tile(np.arange(1, num_routes + 1), num_days),
    'Time': np.random.choice(['Morning', 'Afternoon', 'Evening'], size=num_days * num_routes),
    'PassengerDensity': np.random.randint(10, 100, size=num_days * num_routes) + np.random.normal(0, 10, size=num_days * num_routes).astype(int)
}
df = pd.DataFrame(data)
df['PassengerDensity'] = df['PassengerDensity'].clip(lower=0) #ensure no negative density
# --- 2. Data Cleaning and Feature Engineering ---
# (In a real-world scenario, this would involve handling missing data, outliers, etc.)
#For this example, we'll add a simple feature: DayOfWeek
df['DayOfWeek'] = df['Date'].dt.dayofweek #Monday=0, Sunday=6
# --- 3. Analysis: Linear Regression for Passenger Density Prediction ---
# We'll predict passenger density based on day of week and time of day.  A more sophisticated model would be needed for real-world application.
results = []
for route in df['Route'].unique():
    route_data = df[df['Route'] == route]
    for time in route_data['Time'].unique():
        time_data = route_data[route_data['Time'] == time]
        slope, intercept, r_value, p_value, std_err = linregress(time_data['DayOfWeek'], time_data['PassengerDensity'])
        results.append({'Route': route, 'Time': time, 'Slope': slope, 'Intercept': intercept, 'R-squared': r_value**2})
results_df = pd.DataFrame(results)
# --- 4. Visualization ---
plt.figure(figsize=(12, 6))
for route in df['Route'].unique():
    route_data = df[df['Route'] == route]
    plt.plot(route_data['Date'], route_data['PassengerDensity'], label=f'Route {route}')
plt.xlabel('Date')
plt.ylabel('Passenger Density')
plt.title('Passenger Density Over Time by Route')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
output_filename = 'passenger_density_over_time.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(10, 6))
sns.barplot(x='Time', y='R-squared', hue='Route', data=results_df)
plt.title('R-squared Values for Linear Regression Models')
plt.xlabel('Time of Day')
plt.ylabel('R-squared')
plt.tight_layout()
output_filename2 = 'r_squared_values.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")