from vectornet import VectorNet
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from dataset import GraphDataset
import os
from utils.config import (
    INTERMEDIATE_DATA_DIR,
)
from animator import Animator
from utils.eval import get_eval_metric_results

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Define model 
    in_channels = 8
    out_channels = 60
    sub_graph_layers = 3
    model = VectorNet(in_channels, out_channels, sub_graph_layers).to(device)

    # train parameter
    batch_size = 4096
    epochs = 150
    validation_cnt = 3
    validation_epoch = epochs / validation_cnt  

    decay_lr_every = 50
    lr = 0.01
    decay_lr_factor = 0.8
    max_n_guesses = 1
    horizon = 30
    miss_threshold = 2.0

    # trainer
    trainer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        trainer, step_size=decay_lr_every, gamma=decay_lr_factor)
    
    # loss
    loss = nn.MSELoss(reduction='mean')

    # animator
    animator = Animator(xlabel='epoch', xlim=[1, epochs], ylim=[0.3, 200],legend=['train loss'])

    # data set
    train_dir = os.path.join(INTERMEDIATE_DATA_DIR,'train_intermediate')
    train_dataset = GraphDataset(train_dir)
    train_data_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)

    valid_dir = os.path.join(INTERMEDIATE_DATA_DIR,'val_intermediate')
    valid_dataset = GraphDataset(valid_dir)
    valid_data_iter = DataLoader(valid_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        acc = 0.0
        num_samples = 0
        model.train()
        for data in train_data_iter:
            data.to(device)
            trainer.zero_grad()
            y = data.y.reshape(-1, out_channels)
            y_hat = model(data)
            l = loss(y, y_hat)
            l.backward()

            acc += l.item() * batch_size
            num_samples += y.shape[0]

            trainer.step()
        scheduler.step()
        animator.add(epoch+1, acc/num_samples)
        print(f"epoch {epoch}, acc: {acc/num_samples}, lr: {trainer.state_dict()['param_groups'][0]['lr']: .6f}")

        if (epoch+1) % validation_epoch == 0 or (not (epoch + 1 < epochs)):
            print(f"eval as epoch:{epoch}")
            metrics = get_eval_metric_results(model, valid_data_iter, device, out_channels, max_n_guesses, horizon, miss_threshold)
            curr_minade = metrics["minADE"]
            print(f"minADE:{metrics['minADE']:3f}, minFDE:{metrics['minFDE']:3f}, MissRate:{metrics['MR']:3f}")

            # if curr_minade < best_minade:
            #     best_minade = curr_minade
            #     save_checkpoint(save_dir, model, optimizer, epoch, best_minade, date)

    
    torch.save(model, "vector_net.pt")

