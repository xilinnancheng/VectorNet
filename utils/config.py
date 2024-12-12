RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}
LANE_WIDTH = {"MIA": 3.84, "PIT": 3.97}
VELOCITY_THRESHOLD = 1.0
# Number of timesteps the track should exist to be considered in social context
EXIST_THRESHOLD = 5
# index of the sorted velocity to look at, to call it as stationary
STATIONARY_THRESHOLD = 13
color_dict = {"AGENT": "#d33e4c", "OTHERS": "#00FF00", "AV": "#0000FF"} # agent: red others: yellow av:blue
LANE_RADIUS = 30
OBJ_RADIUS = 30
DATA_DIR = "./mydata"
OBS_LEN = 20  # current time index of sequence
INTERMEDIATE_DATA_DIR = "./interm_data"
