from metocean_api import ts

# Fauskane

wind_wave_ts = ts.TimeSeries(
    lon=5.72684,
    lat=62.5672,
    start_time="1959-01-01",
    end_time="2025-11-30",
    product="NORA3_wind_wave"
)

wind_wave_ts.import_data(save_csv=True, save_nc=False, use_cache=True)

####################################
######### BUOY LOCATIONS ###########
####################################

##### FAUSKANE #####
# wind_wave_ts = ts.TimeSeries(
#     lon=5.72684,
#     lat=62.5672,
#     start_time="1959-01-01",
#     end_time="2025-11-31",
#     product="NORA3_wind_wave"
# )

##### FEDJEOSEN #####
# wind_wave_ts = ts.TimeSeries(
#     lon=4.662131,
#     lat=60.732916,
#     start_time="1959-01-01",
#     end_time="2025-11-31",
#     product="NORA3_wind_wave"
# )

####################################
######### TEST LOCATIONS ###########
####################################

##### LOCATION 1 #####
# wind_wave_ts = ts.TimeSeries(
#     lon=,
#     lat=,
#     start_time="1959-01-01",
#     end_time="2025-11-31",
#     product="NORA3_wind_wave"
# )

##### LOCATION 2 #####
# wind_wave_ts = ts.TimeSeries(
#     lon=,
#     lat=,
#     start_time="1959-01-01",
#     end_time="2025-11-31",
#     product="NORA3_wind_wave"
# )

##### LOCATION 3 #####
# wind_wave_ts = ts.TimeSeries(
#     lon=,
#     lat=,
#     start_time="1959-01-01",
#     end_time="2025-11-31",
#     product="NORA3_wind_wave"
# )