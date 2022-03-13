from sklearn.cluster import SpectralCoclustering
import numpy as np
import pandas as pd

whisky = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@whiskies.csv", index_col=0)
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
correlations = np.array(correlations)

cluster_colors = ['#0173b2', '#de8f05','#029e73', '#d55e00', '#cc78bc', '#ca9161']
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]

region_colours = dict()
for i in range(len(cluster_colors)):
    region_colours[regions[i]] = cluster_colors[i]
# boom
# this can be done simply by
# region_colors = dict(zip(regions, cluster_colors))

# making interactive plots
# ColumnDataSource is a bokeh structure used for defining interactive plotting inputs.





