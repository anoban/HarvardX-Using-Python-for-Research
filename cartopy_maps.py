import pandas as pd
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

pectoral_sandpiper = pd.read_csv("C:\\Users\\Anoba\\Documents\\GPS_DATA\\Arctic shorebird migration tracking study - Pectoral Sandpiper.csv")
pectoral_sandpiper.tail()
pectoral_sandpiper.columns
pectoral_sandpiper.shape  # (862, 10)
coords = pectoral_sandpiper[["location-long","location-lat"]]
coords = coords.rename(columns={
    "location-long":"longitude",
    "location-lat":"latitude"
})

plt.plot(coords.longitude, coords.latitude) # to find the coordinate limits

plt.figure(figsize=(20,20))
ax = plt.axes(projection = ccrs.Mercator())
ax.set_extent((-160, -53, -25, 62))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.plot(coords.longitude, coords.latitude, transform=ccrs.Geodetic(), color="#F35015", linestyle="-", linewidth=1)
plt.title("$Migratory\hspace{1}pattern\hspace{1}of\hspace{1}Arctic\hspace{1}pectoral\hspace{1}sandpiper$", fontsize=25)
plt.savefig("Arctic_sandpiper.jpeg", dpi=1000)
# plt.show()


bylot = pd.read_csv("C:\\Users\\Anoba\\Documents\\GPS_DATA\\Arctic fox Bylot - GPS tracking.csv")
bylot.columns









