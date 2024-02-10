import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def convert_time_to_seconds(df, time_column):
    # Convert the time column to datetime format
    df[time_column] = pd.to_datetime(df[time_column], format='%m/%d/%Y %H:%M:%S.%f')

    # Calculate the time difference in seconds
    df['Time'] = (df[time_column] - df[time_column].min()).dt.total_seconds()

    return df

def reset_time(df, time_column):
    min_time = df[time_column].min()

    # Reset the time column by subtracting the minimum time
    df[time_column] = df[time_column] - min_time

    return df

def get_IC_coefficents(
        df: pd.DataFrame, 
        temperature='Temperature_0 (Collected)', 
        ic_voltage='Voltage_1 (Collected)'
) -> list:
    coefficients = np.polyfit(df[ic_voltage], df[temperature], 1)
    return coefficients

def get_IC_temperatures(df, coefficents, ic_voltage='Voltage_1 (Collected)'):
    a = coefficents[0]
    b = coefficents[1]
    return a*df[ic_voltage]+b

def steinhart_hart_equation(resistance, A, B, C):
    # 1/T = A + Blog(R) + Clog(R)^3
    return  (A + B * np.log(resistance) + C * (np.log(resistance))**3)

def get_steinhart_hart_coefficents(
        df: pd.DataFrame, 
        temperature='Temperature_0 (Collected)', 
        thermistor_voltage='Voltage_0 (Collected)'
) -> list:
    initial_guess = [1, 1, 1]  # You may need to adjust these initial values based on your data
    coefficients, covariance = curve_fit(steinhart_hart_equation, df[thermistor_voltage], 1 / df[temperature], p0=initial_guess)
    return coefficients

def get_thermistor_temperatures(df, coefficents, thermistor_voltage = 'Voltage_0 (Collected)'):
    A,B,C = coefficents[0], coefficents[1], coefficents[2]
    def calculate_T(voltage):
        return 1/steinhart_hart_equation(voltage, A, B, C)
    return df[thermistor_voltage].apply(calculate_T)

def exponential_equation(t,a,tau,b):
    return a * (np.e **(-t/tau)) + b

def get_expoential_coefficents(df, time, column, initial_guess = [1, 1, 1]):
    coefficents, covariance = curve_fit(exponential_equation, df[time], df[column], p0 = initial_guess)
    return tuple(coefficents)

def get_just_time_constant(df, time, column, a, b, tau_initial_guess = [1]):
    coefficents, covariance = curve_fit(lambda t, tau: exponential_equation(t,a,tau,b), df[time], df[column], p0 = tau_initial_guess)
    return coefficents[0]
