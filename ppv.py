import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

d_data = np.array([36.1, 43.2, 48.8, 41.3, 47.2, 53.7])
sd_data1 = d_data[:2] / np.sqrt(5.4),d_data[:-1] / np.sqrt(4.5)
PPV_data1 = np.array([13.7, 9.3, 8.8, 11.8, 9.2, 8.5])


# Function to calculate the fitted curve (power-law equation)
def fitted_curve(sd, a, b):
    return a * sd**(-b)

# Linear regression on logarithmic data
log_sd = np.log10(sd_data)
log_PPV = np.log10(PPV_data)


# Define the power-law function
def power_law(sd, a, b):
    return a * sd**(-b)


popt, pcov = curve_fit(power_law, sd_data, PPV_data,p0=[1000,1.6])
a, b = popt
perr = np.sqrt(np.diag(pcov)) # Standard errors

fitted_sd = np.linspace(min(sd_data), max(sd_data), 100)
fitted_PPV = fitted_curve(fitted_sd, a, b)

# Plotting the results (raw data and logarithmic data with fit curve)

plt.figure(figsize=(10, 6))

# Raw data plot
plt.subplot(121)
plt.scatter(sd_data, PPV_data, label="Raw Data")
plt.plot(fitted_sd, fitted_PPV, label="Fitted Curve", color="red")
plt.xlabel("Scaled Distance (sd)")
plt.ylabel("Peak Particle Velocity (PPV)")
plt.title("Raw Data")
plt.grid(True)

# Logarithmic data plot with fit curve
plt.subplot(122)
plt.scatter(log_sd, log_PPV, label="Logarithmic Data")
plt.plot(np.log10(fitted_sd), np.log10(fitted_PPV), label="Fitted Curve", color="red")
plt.xlabel("log(sd)")
plt.ylabel("log(PPV)")
plt.title("Logarithmic Data with Fitted Curve")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Print results
print(f"Constants: a = {a:.2f} ± {perr[0]:.2f}, b = {b:.2f} ± {perr[1]:.2f}")
