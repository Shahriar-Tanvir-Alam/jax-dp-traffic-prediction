import math
import pandas as pd
# from traffic_prediction.clean import filter_start_hr_to_end_hr

def calc_upper_fence(series):
    q3 = series.quantile(0.75)
    q1 = series.quantile(0.25)
    iqr = q3 - q1
    return q3 + 1.5 * iqr

def create_date_range(start_date: str, end_date: str, start_time: str, end_time: str, freq_date: bool, freq_time: str):
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq_date) 
#     interval = pd.Timedelta(freq_time)
#     time_range = pd.date_range(start=start_time, end=pd.to_datetime(end_time)+interval, freq=freq_time)
    time_range = pd.date_range(start=start_time, end=end_time, freq=freq_time)

    
    print(f'Number of days (frequency={freq_date}): {len(date_range)}')
    print(f'Number of times (frequency={freq_time}): {len(time_range)}')
    result = []

    for date in date_range:
        date_time_range = [date + pd.DateOffset(hours=time.hour, minutes=time.minute, seconds=time.second) for time in time_range]
        result.extend(date_time_range)
        
#     start_time_hour = pd.to_datetime(start_time).time().hour
#     end_time_hour = pd.to_datetime(end_time).time().hour
#     output = pd.DatetimeIndex(filter(lambda x: (x.minute>=start_time_hour and x.hour<=start_time_hour+1) or (x.hour>=end_time_hour-1 and x.hour<=end_time_hour), result))
    output = pd.DatetimeIndex(result)
    
    return output 


# def filter_start_hr_to_end_hr(df, datetime_col, start_hr, end_hr):
#     # assumes datetime column was cast using pd.to_datetime()
#     return df[(df[datetime_col].dt.hour >= pd.to_datetime(start_hr).hour) & (df[datetime_col].dt.hour < pd.to_datetime(end_hr).hour)]


# def aggregate_realtime_speed(df, start_date: str, end_date: str, start_time: str, end_time: str, final_start_time: str, final_end_time: str, freq_date: str ='B', freq_time: str ='15min'):
def aggregate_realtime_speed(df, start_date: str, end_date: str, start_time: str, end_time: str, freq_date: str ='B', freq_time: str ='15min'):
    """ calculate the average speed for each link at each time interval
    """
    date_range = create_date_range(start_date, end_date, start_time, end_time, freq_date, freq_time)

    df['time_bucket'] = pd.cut(df['datetime'], bins=date_range)
    
    df['time_bucket'] = df['time_bucket'].apply(lambda x: x.left)

#     df = filter_start_hr_to_end_hr(df, 'time_bucket', final_start_time, final_end_time)
    
    # series indexed by pandas time interval and a link id (int)
#     aggregated = df.groupby(['time_bucket', 'link_id'])['speed'].mean()
    aggregated = df.groupby(['time_bucket', 'id'])['speed'].mean()
    aggregated_df = aggregated.to_frame()
    aggregated_df.columns = ['speed']
    return aggregated_df


# def aggregate_tlc(df, start_date, end_date, start_time, end_time, final_start_time, final_end_time, neighbors, freq_date: str ='B', freq_time: str ='15min'):
def aggregate_tlc(df, start_date, end_date, start_time, end_time, neighbors, freq_date: str ='B', freq_time: str ='15min'):
    """ calculate the inflow and outflow from each taxi zone at each time interval
        calculate the inflow and outflow speed for each taxi zone at each time interval
        using only trips to neighboring zones
        neigbhors: dict{taxi_zone_id, [taxi_zone_ids]}
    """
    df = df.copy()
    date_range = create_date_range(start_date, end_date, start_time, end_time, freq_date, freq_time)
    
    print(f'Total number of date-times: {len(date_range)}')
          
    df['pickup_time_bucket'] = pd.cut(df['pickup_datetime'], bins=date_range)
    df['dropoff_time_bucket'] = pd.cut(df['dropoff_datetime'], bins=date_range)

    df['pickup_time_bucket'] = df['pickup_time_bucket'].apply(lambda x: x.left)
    df['dropoff_time_bucket'] = df['dropoff_time_bucket'].apply(lambda x: x.left)
    
#     df = filter_start_hr_to_end_hr(df, 'pickup_time_bucket', final_start_time, final_end_time)
#     df = filter_start_hr_to_end_hr(df, 'dropoff_time_bucket', final_start_time, final_end_time)
    
    # calculating the traffic volume for each node
    inflow_volume = df.groupby(['dropoff_time_bucket', 'DOLocationID']).size()
    outflow_volume = df.groupby(['pickup_time_bucket', 'PULocationID']).size()

    aggregated_df = inflow_volume.to_frame()
    aggregated_df.columns = ['inflow_volume']
    aggregated_df.rename_axis(index={'dropoff_time_bucket': 'time_bucket', "DOLocationID": "location_id"}, inplace=True)
    aggregated_df['outflow_volume'] = outflow_volume
    
    print(f"Shape of dataframe: {aggregated_df.shape}")
    return aggregated_df


def find_neighbors(gdf, neighbor_function, region_nodes=None):
    """returns a dictionary mapping a taxi zone LocationID to a list of neighboring LocationIDs"""
    neighbors = {}
    for zone1 in gdf.itertuples():
        if math.isnan(zone1.LocationID): continue
        if region_nodes:
            if (zone1.LocationID not in region_nodes): 
                continue
        touching = []
        for zone2 in gdf.itertuples():
            method_to_call = getattr(zone2.geometry, neighbor_function)
            if zone1.LocationID==zone2.LocationID: 
                continue
            if method_to_call(zone1.geometry) and not math.isnan(zone2.LocationID):
                if region_nodes:
                    if zone2.LocationID not in region_nodes: 
                        continue
                touching.append(zone2.LocationID)
        neighbors[zone1.LocationID] = touching
    return neighbors

def aggregate_feature_values(df_input, datetime_analysis, features=['inflow_volume', 'outflow_volume'], missing_condition=None):
    df = df_input.copy()

    df['time_bucket_col'] = df.index.get_level_values(0)
    
    if datetime_analysis=='time':
        df['time'] = df['time_bucket_col'].apply(lambda x: x.time)
    elif datetime_analysis=='day':
        df['day'] = df['time_bucket_col'].apply(lambda x: x.date())
    elif datetime_analysis=='dayofweek':
        df['dayofweek'] = df['time_bucket_col'].apply(lambda x: x.dayofweek)
        
    df.reset_index(inplace=True)
    
    if missing_condition:
        total_buckets = df.groupby(datetime_analysis)[datetime_analysis].count()
        if missing_condition=='0':
            aggreagted_feature = df.groupby(datetime_analysis)[features].apply(lambda x: (x == 0).sum())
        elif missing_condition=='null':
            aggreagted_feature = df.groupby(datetime_analysis)[features].apply(lambda x: (x.isna()).sum())
        for x in features:
            aggreagted_feature['percent_missing_'+x] = (aggreagted_feature[x] / total_buckets * 100).round(2)
            aggreagted_feature['percent_missing_'+x] = aggreagted_feature['percent_missing_'+x].fillna(0)
    else:  
        aggreagted_feature = df.groupby(datetime_analysis)[features].mean()
        
    return aggreagted_feature

def get_single_rolling_avg_with_neighbors(df, location_id, neighbors, window_size, value_column):
    # Returns the average data of value_column for location_id: 1. average of neighbors data 2. rolling average of window_size times after the neighbors averaging
    # Get the main location's data
    # df.xs: Return cross-section from the Series/DataFrame.
    main_location_data = df.xs(location_id, level='location_id')[[value_column]]
    
    # DataFrames to store neighbor data
    data_frames = [main_location_data]
    
    # Loop over neighbors to collect their data
    available_nodes = set(df.index.get_level_values('location_id').unique())
    for neighbor in neighbors[location_id]:
        if neighbor not in available_nodes: continue
        neighbor_data = df.xs(neighbor, level='location_id')[[value_column]]
        data_frames.append(neighbor_data)
    
    # Concatenate all data frames along time axis 
    combined_data = pd.concat(data_frames, axis=1).mean(axis=1)
    
    # Now calculate the rolling mean on this combined data
    return combined_data.rolling(window=window_size, min_periods=1).mean()

def get_rolling_avg_neighbors(df, window_size, neighbors):
    # Apply the aggregation function for each location
    results = {}
    for location in df.index.get_level_values('location_id').unique():
        results[location] = {
            'inflow_volume': get_single_rolling_avg_with_neighbors(df, location, neighbors, window_size, 'inflow_volume'),
            'outflow_volume': get_single_rolling_avg_with_neighbors(df, location, neighbors, window_size, 'outflow_volume')
        }

    # Combine results into a new DataFrame, keeping location_id in the index
    results_df = pd.concat(
        {location: pd.DataFrame(data) for location, data in results.items()},
        names=['location_id', 'time_bucket']
    )

    # Reset index to turn MultiIndex levels into columns, then set them back as indices
    results_df = results_df.reset_index()
    results_df.set_index(['time_bucket', 'location_id'], inplace=True)

    # Sort the index to ensure data is well-organized (optional but recommended)
    results_df.sort_index(inplace=True)

    display(results_df)
    return results_df