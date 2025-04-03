import torch
import numpy as np
import os
from mydataloader import Prior
from utils import TransformerModel
from setting import D, PriorProcesser

def train_transformer(input_dim, gpu_idx):
    # Set which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    # Setup training configurations
    batch_size = 256
    steps = 10000
    embed_dim = 1024
    depth = 10
    lr = 0.00001

    # Initialize dataset and model
    prior = D(D_num=input_dim, sm=1/16, sw=1/16)
    priorProcesser = PriorProcesser(prior)
    model = TransformerModel(n_dims=priorProcesser.d, n_positions=32,
                             n_embd=embed_dim, n_layer=depth, n_head=8)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        input_, target, _ = priorProcesser.draw_sequences(bs=batch_size, k=32)
        input_ = torch.from_numpy(input_).cuda().float()
        target = torch.from_numpy(target).cuda().float()

        optimizer.zero_grad()
        output = model(input_, target)
        loss = (target - output).square().mean()
        loss.backward()
        optimizer.step()

    print(f"Training Complete for Input Dimension: {input_dim}")
    return model

if __name__ == "__main__":
    # For standalone testing: trains for input_dim=4 on GPU 0
    train_transformer(input_dim=4, gpu_idx=0)