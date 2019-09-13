#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plt
import xlrd
import datetime
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler
from pandas._libs import tslibs 
#Import the data_set
raw_data = pd.read_csv('DataSet/spc_energy_minute_tb_2.csv', header=None, index_col=False)
raw_data.columns = ['Num', 'Addr', 'Time', 'AllEnergy', 'EnergyX',
       'EnergyY', 'EnergyZ']

AllData = raw_data[[ 'Addr','Time','AllEnergy']]

Data_of_one = AllData[AllData['Addr'] == 131073]
Data_of_one = Data_of_one.reset_index(drop=True)

#Using a diction to record the connection between time and Power use
orig_date_power = {}
Time = np.array(pd.to_datetime(Data_of_one['Time']))

#Sort the time series
Time_sorted = np.sort(Time)
AllEnergy = np.array(Data_of_one['AllEnergy'])
num_of_rows = np.size(Time)

#Establish the Connection
for i in range(1, num_of_rows+1):
    date = Time[i-1]
    power = int(AllEnergy[i-1])
    orig_date_power[date] = power
    
AllEnergy_sorted = []
for i in range(0, num_of_rows):
    AllEnergy_sorted.append( orig_date_power[Time_sorted[i]] )
    
#Set the beginning time and end time
start_date = Time_sorted[0]  # 开始时间
end_date = Time_sorted[num_of_rows-1]  # 结束时间
time_point = [start_date]
gen_date_power = {}
date = start_date
i = 0

            
# Set the start time of every month to shorten the running time of enery search;
month_start = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
month_start[0] = np.datetime64("2017-05-01T00:00:00")
month_start[1] = np.datetime64("2017-06-01T00:00:00")
month_start[2] = np.datetime64("2017-07-01T00:00:00")
month_start[3] = np.datetime64("2017-08-01T00:00:00")
month_start[4] = np.datetime64("2017-09-01T00:00:00")
month_start[5] = np.datetime64("2017-10-01T00:00:00")
month_start[6] = np.datetime64("2017-11-01T00:00:00")
month_start[7] = np.datetime64("2017-12-01T00:00:00")
month_start[8] = np.datetime64("2018-01-01T00:00:00")
month_start[9] = np.datetime64("2018-02-01T00:00:00")
month_start[10] = np.datetime64("2018-03-01T00:00:00")
month_start[11] = np.datetime64("2018-04-01T00:00:00")
month_start[12] = np.datetime64("2018-05-01T00:00:00")
month_start[13] = np.datetime64("2018-06-01T00:00:00")
month_start[14] = np.datetime64("2018-07-01T00:00:00")
month_start[15] = np.datetime64("2018-08-01T00:00:00")
month_start[16] = np.datetime64("2018-09-01T00:00:00")
month_start[17] = np.datetime64("2018-10-01T00:00:00")
month_start[18] = np.datetime64("2018-11-01T00:00:00")
month_start[19] = np.datetime64("2018-12-01T00:00:00")

#Define the function get_power() to get the power use if a certain moment;
def get_power( date_cons, month):
    i =  3000*month
    if Time_sorted[i] >= date_cons:
        return AllEnergy_sorted[i]
    while Time_sorted[i] < date_cons:
        if Time_sorted[i+1] > date_cons:
#             print(Time_sorted[i+1])
            return AllEnergy_sorted[i]
        else:
            i = i+1 
#             print(Time_sorted[i])
    return AllEnergy_sorted[i]

#Get the power used monthly;
def get_power_month (month):
    gen_date_power = {}
    date_ini = month_start[month]# Set the Start Time
    date = date_ini
    while date < month_start[month + 1]:
        gen_date_power[np.int64(date) - np.int64(date_ini)] = get_power(date,month)
        if (gen_date_power[np.int64(date) - np.int64(date_ini)] == 0 ):
            gen_date_power[np.int64(date) - np.int64(date_ini)] = gen_date_power[np.int64(date) - np.int64(date_ini) - 3600]
        print(date)
        date = date + np.timedelta64(1,'h')#Update the Time
    Energy_per_hour = {}
    hour = 1
    while hour < len(gen_date_power) :
        Energy_per_hour[hour] = gen_date_power[3600*hour] - gen_date_power[3600*(hour - 1)]
        hour = hour +1
    return Energy_per_hour

def Smoothing(data_month):
    N = int(len(data_month)/24)
    w = {}
    n_time_point = 24
    for minutes in range(n_time_point):
        w[minutes] = {}
        for days in range(0, N):
            w[minutes][days] = data_month[days * n_time_point + minutes+1]
    E = {}
    D = {}
    for keys in w:
        E[keys] = sum(w[keys].values())/len(w[keys])
        D[keys] = sum([(w[keys][days] - E[keys]) * (w[keys][days] - E[keys]) for days in w[keys]]) / len(w[keys])
    rho_list = []
    data_x = 0
    rho_threshold = 0.0003
    result_power = {}
    for times in w:
        for days in w[times]:
            data_x = data_x + 1
            rho = abs(w[times][days] - E[times]) / (D[times]+0.0000000001)
            rho_list.append(rho)
            if rho >= rho_threshold:
                if 0 < days < N-1:
                    w[times][days] = (w[times][days-1] + w[times][days+1]) / 2
                elif days == 0:
                    w[times][days] = (w[times][days] + w[times][days + 1]) / 2
                else:
                    w[times][days] = (w[times][days] + w[times][days - 1]) / 2
            result_power[days * 24 + times] = w[times][days]
    days_list = list(result_power.keys())
    days_list.sort()
    final_power = {}
    for i in days_list:
        final_power[i] = result_power[i]
    return(final_power)


data_month = [];
for i in range(19):
    data_month.append(Smoothing(get_power_month(i)));
    
# Transform the Data form to DataFrame and then output
for i in range(19):
    data_month[i] = pd.DataFrame.from_dict(data_month[i], orient='index')

for i in range(19):
    pd.DataFrame(data_month[i]).to_csv('DataProcessed/Data_month_{}.csv'.format(str(i)))

