import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import contextily as cx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from itertools import chain
import matplotlib.dates as mdates


def plot_taxi_zones(gdf, heatmap_feature, hover_name):
    fig = px.choropleth(gdf,
                    geojson=gdf.geometry.__geo_interface__,
                    locations=gdf.index,
                    color=heatmap_feature,
                    color_continuous_scale='OrRd',
                    hover_name=hover_name)

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def plot_geo_with_back_map(gdf, heatmap_feature, hover_name):
    bbox = gdf.total_bounds
    fig = go.Figure()
    fig.add_trace(go.Choroplethmapbox(geojson=gdf.geometry.__geo_interface__,
                                      locations=gdf.index,
                                      z=gdf[heatmap_feature],
                                      colorscale='OrRd',
                                      colorbar=dict(title=hover_name),
                                      marker_opacity=0.7))

    fig.update_layout(mapbox_style="open-street-map", # carto-positron
                      mapbox_zoom=9,
                      mapbox_center=dict(lat=(bbox[1] + bbox[3]) / 2, lon=(bbox[0] + bbox[2]) / 2))
    fig.show()
    
    
def split_points(idx, row):
    points = row['link_points'].split()
    lats, lons = [], []
    for point in points:
        lat_lon = point.split(',')
        if len(lat_lon) != 2: continue
        # for some reason, sometimes you only get a latitude
        try:
            lat = float(lat_lon[0])
            lon = float(lat_lon[1])
        except ValueError as e:
            # some are not properly formated
            continue
        lats.append(lat)
        lons.append(lon)

    color_opacity = f'rgbd({np.random.randint(0, 255)},{np.random.randint(0, 255)},{np.random.randint(0, 255)},{0.95})'

    data = {'link_id': [idx] * len(lats),
            'color': [color_opacity] * len(lats),
            'latitude': lats,
            'longitude': lons}

    return pd.DataFrame(data)

def plot_links(link_df, zoom=10, height=1000, width=1000, marker_size=6):
    np.random.seed(1)

    new_dfs = []
    for idx, row in link_df.iterrows():
        new_df = split_points(idx, row)
        new_dfs.append(new_df)

    plotting_df = pd.concat(new_dfs, ignore_index=True)

    plotting_df['dummy_size'] = 1

    fig = px.scatter_mapbox(plotting_df, lat="latitude",    lon="longitude", color="color",
                            zoom=zoom, height=height, hover_name='link_id',
                            width=width, size='dummy_size', size_max=marker_size)
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()

# Plot the list of longitude-latitude pairs
def plot_coordinates(lon_lat_array_input, zoom=10, height=1000, width=1000, marker_size=10):
    if type(lon_lat_array_input[0]) is tuple:
        lon_lat_array = np.array([list(x) for x in lon_lat_array_input])
    else:
        lon_lat_array = lon_lat_array_input
        
    df = pd.DataFrame(list(zip(lon_lat_array[:, 0], lon_lat_array[:, 1])), columns =['lat', 'lon'])
    df['dummy_size'] = 1 
    fig = px.scatter_mapbox(df, lat="lat", lon="lon", zoom=zoom, height=height, width=width, \
                            opacity=.8, size='dummy_size', size_max=marker_size)
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()

def plot_rides_by_provider(df):
    license_num_to_provider = {'HV0002': 'Juno', 'HV0003': 'Uber', 'HV0004': 'Via', 'HV0005': 'Lyft'}

    if 'provider' not in df.columns:
        df['provider'] = df['hvfhs_license_num'].map(license_num_to_provider)

    df_grouped = df.groupby(['month', 'provider']).size().reset_index(name='count')

    fig = px.bar(df_grouped, x='month', y='count', color='provider',
             labels={'month': 'Month', 'count': 'Count'},
             title='Number of Rides by Provider each Month')

    fig.update_layout(barmode='group')
    fig.update_traces(textposition='auto')
    fig.show()

    

def plot_trip(tlc_df, zones_gdf, trip_number):
    PULocationID_temp = tlc_df.loc[trip_number, 'PULocationID']
    DOLocationID_temp = tlc_df.loc[trip_number, 'DOLocationID']

    print("trip: \n", tlc_df.loc[trip_number, :])

    print('LocationID:\n')
    display(zones_gdf[(zones_gdf['LocationID']==PULocationID_temp)])
    display(zones_gdf[(zones_gdf['LocationID']==DOLocationID_temp)])

    selected_failed_zones = [PULocationID_temp, DOLocationID_temp]
    
    zones_gdf['selected_regions'] = zones_gdf['LocationID'].isin(selected_failed_zones).astype(int)
    plot_geo_with_back_map(zones_gdf, heatmap_feature='selected_regions', hover_name='LocationID')
    
    
def compare_features_for_trip(tlc_df, zones_gdf, f1, f2, trip_number):
    PULocationID_temp = tlc_df.loc[trip_number, 'PULocationID']
    DOLocationID_temp = tlc_df.loc[trip_number, 'DOLocationID']

    print("trip: \n", tlc_df.loc[trip_number, :])

    print(f1 + ' :\n')
    display(zones_gdf[(zones_gdf[f1]==PULocationID_temp)])
    display(zones_gdf[(zones_gdf[f1]==DOLocationID_temp)])

    print('\n\n'+f2+' :\n')
    display(zones_gdf[(zones_gdf[f2]==PULocationID_temp)])
    display(zones_gdf[(zones_gdf[f2]==DOLocationID_temp)])

    selected_failed_zones = [PULocationID_temp, DOLocationID_temp]
    
    zones_gdf['selected_regions'] = zones_gdf[f1].isin(selected_failed_zones)
    plot_taxi_zones(zones_gdf, heatmap_feature='selected_regions', hover_name=f1)

    zones_gdf['selected_regions'] = zones_gdf[f2].isin(selected_failed_zones)
    plot_taxi_zones(zones_gdf, heatmap_feature='selected_regions', hover_name=f2)
    
def plot_0_values_box_bucket(dfs):
    nan_tlc_dfs = {}
    zones_to_remove = {}

    def calc_upper_fence(series):
        q3 = series.quantile(0.75)
        q1 = series.quantile(0.25)
        iqr = q3 - q1
        return q3 + 1.5 * iqr

    for bucket_size in dfs.keys():
        cur_df = dfs[bucket_size]
        nan_counts_per_link = ((cur_df['inflow_volume'] == 0) & (cur_df['outflow_volume'] == 0)).groupby('location_id').sum().to_frame(name='# temporal buckets missing data')
        total_temporal_buckets = dfs[bucket_size].index.get_level_values(0).nunique()
        nan_counts_per_link['% temporal buckets missing data'] = (nan_counts_per_link['# temporal buckets missing data'] / total_temporal_buckets * 100).round(2)

        nan_tlc_dfs[bucket_size] = nan_counts_per_link
        upper_fence = calc_upper_fence(nan_counts_per_link["% temporal buckets missing data"])

        zones_to_remove[bucket_size] = list(nan_counts_per_link[nan_counts_per_link['% temporal buckets missing data'] > upper_fence].index)

    combined_df = pd.concat(nan_tlc_dfs.values())
    combined_df['Bucket Size'] = [key for key in nan_tlc_dfs.keys() for _ in range(len(nan_tlc_dfs[key]))]
    # would need to do the same for id to give as hover_name info
    fig = px.box(combined_df, x='Bucket Size' ,y="% temporal buckets missing data", title='Statistics over all zones on the # buckets missing data')
    fig.show()
    return nan_counts_per_link
    
    
def plot_0_inflow_outflow_values_bucket(dfs):
    outflow_percent_missing = []
    inflow_percent_missing = []
    both_percent_missing = []

    for bucket_size in dfs.keys():
        feat_df = dfs[bucket_size]

        total_buckets = len(feat_df) # spatial-temporal buckets

        zero_outflow_buckets = (feat_df['outflow_volume'] == 0).sum()
        outflow_percent_missing.append(zero_outflow_buckets/total_buckets * 100)

        zero_inflow_buckets = (feat_df['inflow_volume'] == 0).sum()
        inflow_percent_missing.append(zero_inflow_buckets/total_buckets * 100)

        zero_outflow_inflow_buckets = ((feat_df['outflow_volume'] == 0) & (feat_df['inflow_volume'] == 0)).sum()
        both_percent_missing.append(zero_outflow_inflow_buckets/total_buckets * 100)

    data1 = outflow_percent_missing
    data2 = inflow_percent_missing
    data3 = both_percent_missing

    labels = ['outflow', 'inflow', 'inflow & outflow']
    index_labels = list(dfs.keys())

    group = list(chain([labels[0]] * len(data1), [labels[1]] * len(data2), [labels[2]] * len(data3)))
    index = index_labels  * len(labels)

    # data: [in_speed 5min, in_speed 10min, ... in_speed 30min, out_speed5min, ..., out_speed30min, ...]
    # so we create the group and index accordingly to match. index is the bucket size. group is the feature
    data_dict = {'Data': data1 + data2 + data3,
                 'Feature': group, 
                 'Bucket Size': index}

    fig = px.bar(data_dict, x='Bucket Size', y='Data', color='Feature', barmode='group')

    fig.update_xaxes(title_text='Bucket Size')
    fig.update_yaxes(title_text='% buckets missing data', range=[0,100])
    fig.update_layout(title='Amount of Missing Data by Bucket Size')
    fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': index_labels})

    fig.show()
    
    
def plot_0_surge_values_bucket(dfs):
    o_minus_i_percent_missing = []
    shortage_percent_missing = []
    surplus_percent_missing = []

    for bucket_size in dfs.keys():
        feat_df = dfs[bucket_size]

        total_buckets = len(feat_df) # spatial-temporal buckets

        zero_o_minus_i_buckets = (feat_df['o_minus_i'] == 0).sum()
        o_minus_i_percent_missing.append(zero_o_minus_i_buckets/total_buckets * 100)
        
        zero_shortage_buckets = (feat_df['shortage'] == 0).sum()
        shortage_percent_missing.append(zero_shortage_buckets/total_buckets * 100)
        
        zero_surplus_buckets = (feat_df['surplus'] == 0).sum()
        surplus_percent_missing.append(zero_surplus_buckets/total_buckets * 100)

    data1 = o_minus_i_percent_missing
    data2 = shortage_percent_missing
    data3 = surplus_percent_missing

    labels = ['o_minus_i', 'shortage', 'surplus']
    index_labels = list(dfs.keys())

    group = list(chain([labels[0]] * len(data1), [labels[1]] * len(data2), [labels[2]] * len(data3)))
    index = index_labels  * len(labels)

    # data: [in_speed 5min, in_speed 10min, ... in_speed 30min, out_speed5min, ..., out_speed30min, ...]
    # so we create the group and index accordingly to match. index is the bucket size. group is the feature
    data_dict = {'Data': data1 + data2 + data3,
                 'Feature': group, 
                 'Bucket Size': index}

    fig = px.bar(data_dict, x='Bucket Size', y='Data', color='Feature', barmode='group')

    fig.update_xaxes(title_text='Bucket Size')
    fig.update_yaxes(title_text='% buckets missing data', range=[0,100])
    fig.update_layout(title='Amount of Missing Data by Bucket Size')
    fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': index_labels})

    fig.show()
def plot_missing_values(df_input, datetime_analysis, missing_condition='0', features=['inflow_volume', 'outflow_volume']):
    
    df = df_input.copy()
    df['time_bucket_col'] = df.index.get_level_values(0)
    if datetime_analysis=='time':
        df['time'] = df['time_bucket_col'].apply(lambda x: x.time)
    elif datetime_analysis=='day':
        df['day'] = df['time_bucket_col'].apply(lambda x: x.date())
    elif datetime_analysis=='dayofweek':
        df['dayofweek'] = df['time_bucket_col'].apply(lambda x: x.dayofweek)
    df.reset_index(inplace=True)
    
    if missing_condition=='0':
        missing = df.groupby(datetime_analysis)[features].apply(lambda x: (x == 0).sum())
    elif missing_condition=='null':
        missing = df.groupby(datetime_analysis)[features].apply(lambda x: (x.isna()).sum())
    total_buckets = df.groupby(datetime_analysis)[datetime_analysis].count()

    for x in features:
        missing['percent_missing_'+x] = (missing[x] / total_buckets * 100).round(2)
        missing['percent_missing_'+x] = missing['percent_missing_'+x].fillna(0)

    # Create a dictionary to map integer values to names
    index_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 
                 5: 'Saturday', 6: 'Sunday'}  
    # Replace the number of day of the week with its name 
    if datetime_analysis=='time':
        missing.index = [str(time_obj) for time_obj in missing.index]
    elif datetime_analysis=='day':
        missing.index = [str(time_obj)+", "+index_mapping[time_obj.weekday()] for time_obj in missing.index]
    elif datetime_analysis=='dayofweek':
        missing.index = missing.index.map(index_mapping)
       
    
    ax = missing.plot(y=['percent_missing_'+x for x in features], label=features)
    # Set y-axis and x-axis titles
    ax.set_ylabel('% of '+missing_condition+' values')
    ax.set_xlabel(datetime_analysis)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  
    # Add a legend
    plt.legend()
    plt.show()
    
    
def plot_feature_values(aggreagted_feature, datetime_analysis, features=['inflow_volume', 'outflow_volume'], missing_condition = None, legend = True, save_address = None):
    # Create a dictionary to map integer values to names
    index_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 
                 5: 'Saturday', 6: 'Sunday'}  
    # Replace the number of day of the week with its name 
    if datetime_analysis=='time':
        aggreagted_feature.index = [str(time_obj) for time_obj in aggreagted_feature.index]
    elif datetime_analysis=='day':
        aggreagted_feature.index = [str(time_obj)+", "+index_mapping[time_obj.weekday()] for time_obj in aggreagted_feature.index]
    elif datetime_analysis=='dayofweek':
        aggreagted_feature.index = aggreagted_feature.index.map(index_mapping)
    
    # Set y-axis and x-axis titles
    if missing_condition:
        ax = aggreagted_feature.plot(y=['percent_missing_'+x for x in features], label=features)
        if missing_condition=='0':
            ax.set_ylabel('% of '+missing_condition+' values')
        elif missing_condition=='null':
            ax.set_ylabel('% of '+missing_condition+' values')
    else:
        ax = aggreagted_feature.plot(y=features, label=features, linewidth=5)
        ax.set_ylabel('Average of feature values')
    ax.set_xlabel(datetime_analysis)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  
    # Add a legend
    if legend:
        plt.legend()
    else:
        ax.get_legend().remove()
    if save_address:
        plt.savefig(save_address)
    plt.show()
    
    return aggreagted_feature



def plot_feature_values_for_comparison(aggreagted_feature, 
                                       datetime_analysis, 
                                       features1=['inflow_volume', 'outflow_volume'], 
                                       features2=['speed'], 
                                       missing_condition=None, 
                                       legend=True, 
                                       save_address=None,
                                       x_tick_counter = 12
                                      ):
    # Create a dictionary to map integer values to names
    index_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 
                     5: 'Saturday', 6: 'Sunday'}  
    
    # Replace the number of day of the week with its name 
    if datetime_analysis == 'time':
        aggreagted_feature.index = [str(time_obj) for time_obj in aggreagted_feature.index]
    elif datetime_analysis == 'day':
        aggreagted_feature.index = [str(time_obj)+", "+index_mapping[time_obj.weekday()] for time_obj in aggreagted_feature.index]
#     elif datetime_analysis == 'dayofweek':
#         aggreagted_feature.index = aggreagted_feature.index.map(index_mapping)
    
    # Plot the first set of features on the primary y-axis
    fig, ax1 = plt.subplots()
    aggreagted_feature[features1].plot(ax=ax1, linewidth=5)
    ax1.set_ylabel('Average of Feature 1 Values')
    ax1.set_xlabel(datetime_analysis)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    
    # Set x-ticks to show every nth tick
    if datetime_analysis == 'time':
        ticks = list(range(len(aggreagted_feature.index)))  # Generate indices for x-ticks
        tick_labels = aggreagted_feature.index  # Corresponding labels from the DataFrame
        ax1.set_xticks(ticks[::x_tick_counter])  # Set ticks every 12th index
        ax1.set_xticklabels(tick_labels[::x_tick_counter], rotation=90)  # Set labels for the ticks

    # Plot the second set of features on the secondary y-axis
    ax2 = ax1.twinx()  # Shares the same x-axis
    aggreagted_feature[features2].plot(ax=ax2, linewidth=5, color='red')  # Add color for distinction
    ax2.set_ylabel('Average of Feature 2 Values')
    
    # Add legends to both axes
    if legend:
        ax1.legend(features1, loc='upper left')
        ax2.legend(features2, loc='upper right')
    else:
        ax1.get_legend().remove()
        ax2.get_legend().remove()
    
    # Save the figure if a path is provided
    if save_address:
        plt.savefig(save_address)
    plt.show()
    
    
def get_mapbox_colors(remove_light_colors=False):
    color_choices_str = "aliceblue,  aqua, aquamarine, azure,\
                beige, bisque, black, blanchedalmond, blue,\
                blueviolet, brown, burlywood, cadetblue,\
                chartreuse, chocolate, coral, cornflowerblue,\
                cornsilk, crimson, cyan, darkblue, darkcyan,\
                darkgoldenrod, darkgray, darkgrey, darkgreen,\
                darkkhaki, darkmagenta, darkolivegreen, darkorange,\
                darkorchid, darkred, darksalmon, darkseagreen,\
                darkslateblue, darkslategray, darkslategrey,\
                darkturquoise, darkviolet, deeppink, deepskyblue,\
                dimgray, dimgrey, dodgerblue, firebrick,\
                forestgreen, fuchsia, gainsboro,\
                gold, goldenrod, gray, grey, green,\
                greenyellow, honeydew, hotpink, indianred, indigo,\
                ivory, khaki, lavender, lavenderblush, lawngreen,\
                lemonchiffon, lime, limegreen,\
                linen, magenta, maroon, mediumaquamarine,\
                mediumblue, mediumorchid, mediumpurple,\
                mediumseagreen, mediumslateblue, mediumspringgreen,\
                mediumturquoise, mediumvioletred, midnightblue,\
                mintcream, mistyrose, moccasin, navy,\
                oldlace, olive, olivedrab, orange, orangered,\
                orchid, palegoldenrod, palegreen, paleturquoise,\
                palevioletred, papayawhip, peachpuff, peru, pink,\
                plum, powderblue, purple, red, rosybrown,\
                royalblue, rebeccapurple, saddlebrown, salmon,\
                sandybrown, seagreen, seashell, sienna, silver,\
                skyblue, slateblue, slategray, slategrey, snow,\
                springgreen, tan, teal, thistle, tomato,\
                turquoise, violet"

    colors = color_choices_str.replace(" ", "").split(",")
    if remove_light_colors:
        modified_colors = [color for color in colors if color not in ['seashell', 'honeydew', 'snow', 'lightcoral', 'mintcream',\
                                                                     'lightcyan', 'oldlace', 'mistyrose', 'whitesmoke', \
                                                                      'ghostwhite', 'ivory', 'aliceblue', 'lavender', \
                                                                      'floralwhite', 'antiquewhite', 'lavenderblush', 'white', \
                                                                     'papayawhip', 'palegoldenrod', 'aqua', 'aquamarine', \
                                                                      'azure', 'thistle', 'gainsboro', 'silver']]
    else:
        modified_colors = colors
    return modified_colors


def plot_ci(model_names, models_to_skip, mean_losses, ci_lows, ci_highs, save_address = None):
    # Filter data to exclude the skipped model
    filtered_indices = [i for i, name in enumerate(model_names) if name not in models_to_skip]
    filtered_model_names = [model_names[i] for i in filtered_indices]
    filtered_mean_losses = [mean_losses[i] for i in filtered_indices]
    filtered_ci_lows = [ci_lows[i] for i in filtered_indices]
    filtered_ci_highs = [ci_highs[i] for i in filtered_indices]

    # Plot the error bars with means connected by a line
    x_positions = range(len(filtered_model_names))
    plt.errorbar(
        x_positions, 
        filtered_mean_losses, 
        yerr=[
            [mean - low for mean, low in zip(filtered_mean_losses, filtered_ci_lows)],  # Lower error
            [high - mean for mean, high in zip(filtered_mean_losses, filtered_ci_highs)]  # Upper error
        ],
        fmt='o', 
        color='red', 
        capsize=10, 
        label='Mean Loss with CI',
        linewidth=6,
        markersize=10,
        markeredgecolor='black',
    )
    plt.plot(x_positions, filtered_mean_losses, color='blue', linestyle='-', linewidth=6, label='Mean Loss Line')

    # Add labels and legend
    plt.xticks(x_positions, filtered_model_names, rotation=90)
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('Error Bars with Connected Mean Line')
    plt.legend(fontsize=14)
    plt.grid(True)

    # Save the plot
    if save_address:
        plt.savefig(save_address)

    # Display the plot
    plt.show()