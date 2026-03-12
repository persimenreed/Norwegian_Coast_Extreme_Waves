from metocean_api import ts


##### Stavanger #####
wind_wave_ts = ts.TimeSeries(
    lon=5.4146939,
    lat=58.9141135,
    start_time="1969-09-01",
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
#     start_time="1969-09-01",
#     end_time="2025-11-30",
#     product="NORA3_wind_wave"
# )

##### FEDJEOSEN #####
# wind_wave_ts = ts.TimeSeries(
#     lon=4.662131,
#     lat=60.732916,
#     start_time="1969-09-01",
#     end_time="2025-11-30",
#     product="NORA3_wind_wave"
# )

##### VESTFJORDEN #####
# wind_wave_ts = ts.TimeSeries(
#     lon=15.477469,
#     lat=68.2307205,
#     start_time="1969-09-01",
#     end_time="2025-11-30",
#     product="NORA3_wind_wave"
# )

####################################
######### TEST LOCATIONS ###########
####################################

##### Kristiansund #####
# wind_wave_ts = ts.TimeSeries(
#     lon=7.570590,
#     lat=63.189756,
#     start_time="1969-09-01",
#     end_time="2025-11-30",
#     product="NORA3_wind_wave"
# )

##### Bergen #####
# wind_wave_ts = ts.TimeSeries(
#     lon=4.8303750,
#     lat=60.3809388,
#     start_time="1969-09-01",
#     end_time="2025-11-30",
#     product="NORA3_wind_wave"
# )

##### Stavanger #####
# wind_wave_ts = ts.TimeSeries(
#     lon=5.4146939,
#     lat=58.9141135,
#     start_time="1969-09-01",
#     end_time="2025-11-30",
#     product="NORA3_wind_wave"
# )