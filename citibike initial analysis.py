# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:32:50 2020

@author: Andrew
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv(r'C:\Users\Andrew\Documents\PythonScripts\Bike Project\2019 June July Citibike Data.csv', index_col=0)

#creating list of stations, lat long, and visits
end_unique = df[['end station name','end station latitude','end station longitude']].drop_duplicates()
end_unique.set_index('end station name',inplace=True)
end_stations = df['end station name'].value_counts().sort_values(ascending=False)
end_unique = end_unique.join(end_stations).dropna().reset_index()
end_unique.columns = ['station name','latitude','longitude','visits']

start_unique = df[['start station name','start station latitude','start station longitude']].drop_duplicates()
start_unique.set_index('start station name',inplace=True)
start_stations = df['start station name'].value_counts().sort_values(ascending=False)
start_unique = start_unique.join(start_stations).dropna().reset_index()
start_unique.columns = ['station name','latitude','longitude','visits']

#create basemap for manhattan
import geopandas as gpd
from shapely.geometry import Point, LineString
import plotly.express as px
import networkx as nx
import osmnx as ox
from plotly.offline import plot
ox.config(log_console=True, use_cache=True)

def create_graph(loc, dist, transport_mode, loc_type="address"):
    """Transport mode = ‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’"""
    if loc_type == "address":
        G = ox.graph_from_address(loc, dist=dist, network_type=transport_mode)
    elif loc_type == "points":
        G = ox.graph_from_point(loc, dist=dist, network_type=transport_mode )
    return G

G = create_graph("New York City", 10000, "bike")
ox.plot_graph(G)

import math
#create haversine filter function that can be applied to a column
def within_x_miles(start_station, other_station,distance_threshold):
    R = 6373.0 #radius of the Earth
        
    lat1 = math.radians(start_unique[start_unique["station name"]==start_station]['latitude'])     
    lon1 = math.radians(start_unique[start_unique["station name"]==start_station]['longitude'])     
    
    #fastest way to filter? since it needs to check distance.                                  
    lat2 = math.radians(end_unique[end_unique["station name"]==other_station]['latitude'])     
    lon2 = math.radians(end_unique[end_unique["station name"]==other_station]['longitude'])     

    #change in coordinates                                 
    dlon = lon2 - lon1      
    dlat = lat2 - lat1

    #Haversine formula    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    #convert from kilometers to miles
    distance = distance/1.603
    
    #multiplier to account for actual route distance
    distance = distance*1.1

    if distance < distance_threshold:
        return True
    else:
        return False

#find random path after haversine filter
def pathfinder(start_station, miles):
    possible_ends = end_unique[end_unique["station name"].apply(lambda x: within_x_miles(start_station, x, miles))]
    
    idx = np.random.randint(5) #select one of top five most visited stations
    
    start = (start_unique[start_unique["station name"]==start_station]['latitude'].iloc[0]
             ,start_unique[start_unique["station name"]==start_station]['longitude'].iloc[0])
    end = (possible_ends.iloc[idx,1],possible_ends.iloc[idx,2])
    start_node = ox.get_nearest_node(G, start) 
    end_node = ox.get_nearest_node(G, end)
    route = nx.shortest_path(G, start_node, end_node, weight='travel_time')
    return route, possible_ends.iloc[idx,:], start_node, end_node

# possible_ends["coordinates"] = list(zip(possible_ends['latitude'], possible_ends['longitude']))
# possible_ends["coordinates"] = possible_ends["coordinates"].apply(lambda x: ox.get_nearest_node(G, x))

#create total route
routes = []
number_stops = 4 #actually +1 because of 0 
nodes = pd.DataFrame(columns=["start_node","end_node"],index=np.arange(number_stops+1))
stations_visited = end_unique[end_unique['station name']=='Scholes St & Manhattan Ave']

i=0 #go for 8 miles?
start_station = 'Scholes St & Manhattan Ave'
while i < number_stops+1:
    print("station number {}: {}".format(i,start_station))
    route_addition, next_station, start_node, end_node= pathfinder(start_station, miles=2)
    
    #saving data
    stations_visited = stations_visited.append(next_station)     
    nodes["start_node"][i]=start_node
    nodes["end_node"][i]=end_node
    routes.append(route_addition)
    
    #set next station in loop
    start_station = str(next_station["station name"])
    i+=1
    
#plot route with osmnx
fig, ax = ox.plot_graph_routes(G, routes, route_color='b', node_size=0)

#conversion to mapbox thanks to https://medium.com/@shakasom/routing-street-networks-find-your-way-with-python-9ba498147342
def create_line_df(routes):
    #create line df for each route
    node_start = []
    node_end = []
    X_to = []
    Y_to = []
    X_from = []
    Y_from = []
    length = []
    
    for u, v in zip(route[:-1], route[1:]):
        node_start.append(u)
        node_end.append(v)
        length.append(round(G.edges[(u, v, 0)]['length']))
        X_from.append(G.nodes[u]['x'])
        Y_from.append(G.nodes[u]['y'])
        X_to.append(G.nodes[v]['x'])
        Y_to.append(G.nodes[v]['y'])
    
    plot_df = pd.DataFrame(list(zip(node_start, node_end, X_from, Y_from,  X_to, Y_to, length)), 
                   columns =["node_start", "node_end", "X_from", "Y_from",  "X_to", "Y_to", "length"]) 
    # plot_df.reset_index(inplace=True)
    return plot_df

def create_line_gdf(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X_from, df.Y_from))
    gdf['geometry_to'] = [Point(xy) for xy in zip(gdf.X_to, gdf.Y_to)]
    gdf['line'] = gdf.apply(lambda row: LineString([row['geometry_to'], row['geometry']]), axis=1)
    line_gdf = gdf[['node_start','node_end','length', 'line']].set_geometry('line')
    return line_gdf

plot_df = pd.DataFrame(columns =["node_start", "node_end", "X_from", "Y_from",  "X_to", "Y_to", "length"]) 

for route in routes:
    temp_plot_df = create_line_df(route)
    plot_df = pd.concat([plot_df,temp_plot_df])
    
plot_df.reset_index(inplace=True,drop=True)
plot_df.reset_index(inplace=True) #doing twice to get column called index of actual index

line_gdf = create_line_gdf(plot_df)

start = plot_df[plot_df['node_start'] == nodes['start_node'][0]]
stop1 = plot_df[plot_df['node_end'] == nodes['end_node'][1]]
stop2 = plot_df[plot_df['node_end'] == nodes['end_node'][2]]
stop3 = plot_df[plot_df['node_end'] == nodes['end_node'][3]]
end = plot_df[plot_df['node_end'] == nodes['end_node'].iloc[-1]]

#create unique set
px.set_mapbox_access_token("pk.eyJ1Ijoic2hha2Fzb20iLCJhIjoiY2plMWg1NGFpMXZ5NjJxbjhlM2ttN3AwbiJ9.RtGYHmreKiyBfHuElgYq_w")
fig = px.scatter_mapbox(plot_df, lon= "X_from", lat="Y_from", zoom=13, width=1000, height=800, animation_frame='index',mapbox_style="dark")
fig.data[0].marker = dict(size = 12, color="black")

#first point
fig.add_trace(px.scatter_mapbox(start, lon= "X_from", lat="Y_from").data[0])
fig.data[1].marker = dict(size = 15, color="red")

#mid points
fig.add_trace(px.scatter_mapbox(stop1, lon= "X_from", lat="Y_from").data[0])
fig.data[2].marker = dict(size = 15, color="green")
fig.add_trace(px.scatter_mapbox(stop2, lon= "X_from", lat="Y_from").data[0])
fig.data[3].marker = dict(size = 15, color="green")
fig.add_trace(px.scatter_mapbox(stop3, lon= "X_from", lat="Y_from").data[0])
fig.data[4].marker = dict(size = 15, color="green")

#last point
fig.add_trace(px.scatter_mapbox(end, lon= "X_from", lat="Y_from").data[0])
fig.data[5].marker = dict(size = 15, color="red")
fig.add_trace(px.line_mapbox(plot_df, lon= "X_from", lat="Y_from").data[0])

plot(fig)
