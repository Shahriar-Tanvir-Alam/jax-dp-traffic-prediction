import datetime as dt
from functools import wraps
import os
import pandas as pd
import plotly.express as px
import geopandas as gpd


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        start_shape = args[0].shape # assume df is always 1st arg of a pandas pipe fn
        result = func(*args, **kwargs) # assume pandas pipe fn always returns a df

        result_shape = result.shape
        time_taken = str(dt.datetime.now() - tic)

        rows = result_shape[0] - start_shape[0]
        cols = result_shape[1] - start_shape[1]
        print(f"step {func.__name__}: {start_shape} --> {result_shape} took {time_taken}s")
        prefix = "+" if cols >= 0 else ""
        print(f"\t {rows} datapoints \t  {prefix}{cols} features")
        return result
    return wrapper

@log_step
def start_pipeline(df):
    # BUG: causes the kernel to crash for large dataframes
    return df.copy()

@log_step
def keep_cols(df, cols):
    return df.loc[:, cols]

@log_step
def cast_datetime(df, col, fmt=None):
    if fmt is not None:
        df[col] = pd.to_datetime(df[col], format=fmt)
    else:
        df[col] = pd.to_datetime(df[col])
    return df

@log_step
def filter_weekdays(df, datetime_col):
    # assumes datetime column was cast using pd.to_datetime()
    # Week starts on Monday as 0 and ends with Sunday as 6
    # Business days = {Monday: 0, Tuesday: 1, Wednesday: 2, Thursday: 3, Friday: 4}
    return df[df[datetime_col].dt.weekday <= 4]

@log_step
def filter_start_hr_to_end_hr(df, datetime_col, start_hr, end_hr):
    # assumes datetime column was cast using pd.to_datetime()
    return df[(df[datetime_col].dt.hour >= start_hr) & (df[datetime_col].dt.hour < end_hr)]

@log_step
def remove_zeros(df, feature):
    return df[df[feature] > 0.0]

@log_step
def rename_feature(df, curr_feature_name, new_feature_name):
    return df.rename(columns={curr_feature_name: new_feature_name})

################################### TLC Dataset specific ###################################
@log_step
def filter_region_tlc(df_tlc, regions):
#     zones = pd.read_csv('../data/tlc_nyc/taxi_zone_lookup_coordinates.csv')
#     borough_zones = zones[zones['Borough'] == region]['LocationID']
    zones_gdf = gpd.read_file('../data/tlc_nyc/taxi_zones/taxi_zones_WGS84.shp')
    zones_gdf['LocationID'] = zones_gdf['LocationID'].apply(int)
#     borough_zones = zones_gdf[zones_gdf['borough'] == region]['LocationID']
    borough_zones = zones_gdf[zones_gdf['borough'].isin(regions)]['LocationID']
    return df_tlc[(df_tlc['PULocationID'].isin(borough_zones)) & (df_tlc['DOLocationID'].isin(borough_zones))]

@log_step
def get_trip_time(df):
    df['trip_time'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()
    return df

@log_step
def check_missing(df, fhvhv=True):
    zero_miles_time = ((df['trip_miles'] == 0.0) & (df['trip_time'] == 0)).sum()
    max_seconds = 2 * 60 * 60
    n_rows = df.shape[0]
    print('# trips with no miles & no time: ', zero_miles_time)
    print('# trips with no miles but time: ', (df['trip_miles'] == 0.0).sum() - zero_miles_time)
    print('# trips with no time but miles: ', (df['trip_time'] == 0).sum() - zero_miles_time)
    print('# trips with no PU OR DO: ', ((df['PULocationID'].isna()) | (df['DOLocationID'].isna())).sum())
    print('# trips with PU after DO: ', (df['pickup_datetime'] > df['dropoff_datetime']).sum())
    print('# trips with 2+ hr trip: ', (df['trip_time'] > max_seconds).sum())
    if fhvhv:
        print('# trips with (Juno, Uber, Via, Lyft): ', (df['hvfhs_license_num'] == 'HV0002').sum(), 
                                                         (df['hvfhs_license_num'] == 'HV0003').sum(), 
                                                         (df['hvfhs_license_num'] == 'HV0004').sum(), 
                                                         (df['hvfhs_license_num'] == 'HV0005').sum())
#     print('# trips with nonpositive base passenger fare: ', (df['base_passenger_fare'] <= 0).sum())
    print('# trips with PU=DO: ', (df['PULocationID'] == df['DOLocationID']).sum())
    
    print('% trips with no miles & no time: ', '{0:.4f}'.format(zero_miles_time/n_rows*100))
    print('% trips with no miles but time: ', '{0:.4f}'.format(((df['trip_miles'] == 0.0).sum() - zero_miles_time)/n_rows*100))
    print('% trips with no time but miles: ', '{0:.4f}'.format(((df['trip_time'] == 0).sum() - zero_miles_time)/n_rows*100))
    print('% trips with no PU OR DO: ', '{0:.4f}'.format(((df['PULocationID'].isna()) | (df['DOLocationID'].isna())).sum()//n_rows*100))
    print('% trips with PU after DO: ', '{0:.4f}'.format((df['pickup_datetime'] > df['dropoff_datetime']).sum()/n_rows*100))
    print('% trips with 2+ hr trip: ', '{0:.4f}'.format((df['trip_time'] > max_seconds).sum()/n_rows*100))
    if fhvhv:
        print('% trips with (Juno, Uber, Via, Lyft): ', '{0:.4f}'.format((df['hvfhs_license_num'] == 'HV0002').sum()/n_rows*100), 
                                                     '{0:.4f}'.format((df['hvfhs_license_num'] == 'HV0003').sum()/n_rows*100), 
                                                     '{0:.4f}'.format((df['hvfhs_license_num'] == 'HV0004').sum()/n_rows*100), 
                                                     '{0:.4f}'.format((df['hvfhs_license_num'] == 'HV0005').sum()/n_rows*100))
#     print('% trips with nonpositive base passenger fare: ', (df['base_passenger_fare'] <= 0).sum()/n_rows)
    print('% trips with PU=DO: ', '{0:.4f}'.format((df['PULocationID'] == df['DOLocationID']).sum()/n_rows*100))
    return df

# @log_step
# def check_missing_yellow(df):
#     zero_miles = ((df['trip_distance'] == 0.0)).sum()
#     n_rows = df.shape[0]
#     print('# trips with 0 miles: ', zero_miles)
#     print('# trips with no PU OR DO: ', ((df['PULocationID'].isna()) | (df['DOLocationID'].isna())).sum())
#     print('# trips with PU after DO: ', (df['tpep_pickup_datetime'] > df['tpep_dropoff_datetime']).sum())
#     print('# trips with PU=DO: ', (df['PULocationID'] == df['DOLocationID']).sum())
    
#     print('% trips with no miles: ', '{0:.4f}'.format(zero_miles/n_rows*100))
#     print('% trips with no PU OR DO: ', '{0:.4f}'.format(((df['PULocationID'].isna()) | (df['DOLocationID'].isna())).sum()//n_rows*100))
#     print('% trips with PU after DO: ', '{0:.4f}'.format((df['tpep_pickup_datetime'] > df['tpep_dropoff_datetime']).sum()/n_rows*100))
#     print('% trips with PU=DO: ', '{0:.4f}'.format((df['PULocationID'] == df['DOLocationID']).sum()/n_rows*100))
#     return df

@log_step
def remove_no_pu_du_loc(df):
    return df[~(df['PULocationID'].isna()) & ~(df['DOLocationID'].isna())]

@log_step
def remove_dropoff_before_pickup(df, pu_t_label='pickup_datetime', do_t_label='dropoff_datetime'):
    return df[df[pu_t_label] < df[do_t_label]]

@log_step
def remove_2hr_plus_trips(df):
    max_seconds = 2 * 60 * 60
    return df[df['trip_time'] <= max_seconds]

@log_step
def remove_app_trips(df, app):
    return df[df['app'] != app]

@log_step
def replace_license_with_app(df):
    # Resource: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf
    license_num_to_provider = {'HV0002': 'Juno', 'HV0003': 'Uber', 'HV0004': 'Via', 'HV0005': 'Lyft'}

    df['app'] = df['hvfhs_license_num'].map(license_num_to_provider)
    df.drop(columns='hvfhs_license_num', inplace=True)

    return df

@log_step
def remove_nonpositive_base_passenger_fare(df):
    return df[df['base_passenger_fare'] <= 0]

################################### Realtime Dataset specific ###################################

@log_step
def filter_region_realtime(df_realtime, regions):
    return df_realtime[df_realtime['BOROUGH'].isin(regions)]

@log_step
def remove_invalid_status(df_realtime):
    # also remove the status column afterwards
    clean_df = df_realtime[df_realtime['STATUS'] == 0]
    clean_df.drop(columns='STATUS', inplace=True)
    return clean_df
