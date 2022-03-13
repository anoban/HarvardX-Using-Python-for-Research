import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

birds = pd.read_csv("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\bird_tracking.csv")
date = pd.Series([d[:-3] for d in birds.date_time])
timestamp = pd.Series([datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date])
birds["timestamp"] = timestamp
birds = birds.drop(["Unnamed: 0", "date_time"], axis=1)
eric = birds[birds.bird_name == "Eric"]
nico = birds[birds.bird_name == "Nico"]
sanne = birds[birds.bird_name=="Sanne"]

cumulative_flight_time = dict()  # voila
for D,d in (["Eric",eric],["Nico", nico],["Sanne", sanne]):
    cumulative_flight_time[D] = list([t - d.timestamp[d.timestamp.index[0]] for t in d.timestamp])

cumulative_flight_time.keys()  # dict_keys(['Eric', 'Nico', 'Sanne'])
# bravo!!!

flight_days = dict()  # you nailed it!!
for gull in ("Eric","Nico","Sanne"):
    flight_days[gull] = np.array(cumulative_flight_time[gull])/datetime.timedelta(days=1)
# bravo!!

