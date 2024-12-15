from utils.feature_utils import (
    compute_feature_for_one_seq,
    encode_features,
    save_features,
)
from argoverse.data_loading.argoverse_forecasting_loader import (
    ArgoverseForecastingLoader,
)
from argoverse.map_representation.map_api import ArgoverseMap
import os
from utils.config import (
    DATA_DIR,
    OBS_LEN,
    LANE_RADIUS,
    OBJ_RADIUS,
    INTERMEDIATE_DATA_DIR,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    am = ArgoverseMap()

    target_viz_index = 5

    for folder in os.listdir(DATA_DIR):
        arg_for_loader = ArgoverseForecastingLoader(os.path.join(DATA_DIR, folder))

        norm_center = {}
        index = 0
        for name in tqdm(arg_for_loader.seq_list):
            index = index + 1
            afl_ = arg_for_loader.get(name)
            path, name = os.path.split(name)
            name, ext = os.path.splitext(name)

            agent_feature, obj_feature_list, lane_feature_list, norm_center = (
                compute_feature_for_one_seq(
                    afl_.seq_df,
                    am,
                    OBS_LEN,
                    LANE_RADIUS,
                    OBJ_RADIUS,
                    viz=index == target_viz_index,
                    mode="nearby",
                )
            )

            df = encode_features(agent_feature, obj_feature_list, lane_feature_list)
            save_features(
                df,
                name,
                os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}_intermediate"),
            )
