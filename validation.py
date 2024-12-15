import torch
import os
from torch_geometric.data import DataLoader
from dataset import GraphDataset, get_data_path_ls
from utils.viz_utils import show_predict_result, show_pred_and_gt
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import (
    INTERMEDIATE_DATA_DIR,
)

if __name__ == "__main__":
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load("vector_net.pt")
        model.eval()
        out_channels = 60

        valid_dir = os.path.join(INTERMEDIATE_DATA_DIR, "val_intermediate")
        valid_dataset = GraphDataset(valid_dir)

        viz_batch_size = 1
        viz_valid_data_iter = DataLoader(valid_dataset, batch_size=viz_batch_size)

        raw_data_path_ls = sorted(get_data_path_ls(valid_dir))
        index = 0
        for data in viz_valid_data_iter:
            data.to(device)
            pred_y = model(data)
            y = data.y.reshape(-1, out_channels)
            torch.set_printoptions(sci_mode=False, precision=3)
            pred_y = pred_y.to(torch.device("cpu"))
            y = y.to(torch.device("cpu"))

            raw_data = pd.read_pickle(raw_data_path_ls[index])
            show_predict_result(raw_data, pred_y, y, raw_data["TARJ_LEN"].values[0])
            index += viz_batch_size
        plt.show()
