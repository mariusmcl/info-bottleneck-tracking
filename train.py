from numpy import mod
import numpy as np
from numpy.lib.utils import info
import torch
import argparse

from activation_trackers import ActivationTracker
from models import get_and_create_model
from data_pipelines import get_data_and_indices
from helpers import make_sure_path_exists
import time
import json
import dill



def train(model, tracker, optimizer, criterion, data_, split_indices, num_epochs, device, verbose=True, model_name=None):
    losses, val_accuracies, train_accuracies = [], [], []
    for key in split_indices:
        split_indices[key] = split_indices[key].to(device)
    data_.x = data_.x.to(device)
    data_.edge_index = data_.edge_index.to(device)
    data_.y = data_.y.to(device)

    for epoch in range(1, num_epochs):
        tracker.register_new_epoch(['act1', 'act2', 'act3'])
        model.train()
        optimizer.zero_grad()

        if model_name == "MLP":
            out = model(data_.x)
        else:
            out = model(data_.x, data_.edge_index)
        train_loss = criterion(out[split_indices["train"]], data_.y[split_indices["train"]])

        train_loss.backward()
        tracker.save()
        losses.append(train_loss.cpu().item())
        optimizer.step()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct = pred[split_indices["train"]] == data_.y[split_indices["train"]]
            acc = int(correct.sum()) / pred[split_indices["train"]].shape[0]
            train_accuracies.append(acc)

            val_correct = pred[split_indices["valid"]] == data_.y[split_indices["valid"]]
            val_acc = int(val_correct.sum()) / pred[split_indices["valid"]].shape[0]
            val_accuracies.append(val_acc)
        
        if verbose:
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Valacc{val_acc:.4f}, Trainacc {acc:.4f}')

    return losses, val_accuracies, train_accuracies


if __name__ == "__main__":

    model_choices = set(("GCN", "GAT", "GIN", "MLP"))
    dataset_choices = set(("Cora", "arxiv", "products"))

    parser = argparse.ArgumentParser(description="Trains a chosen model on a dataset")
    parser.add_argument("--model", default="GCN", choices=model_choices)
    parser.add_argument("--dataset", default="Cora", choices=dataset_choices)
    parser.add_argument("--hidden_dim", default=100, type=int, help="Number of neurons in the hidden layers")
    parser.add_argument("--activation", default="relu", help="Activation to be used for hidden layers in the network")
    parser.add_argument("--epochs", default=100, help="Number of epochs to train with gradient descent. Note that we use the entire dataset for each gradient update")
    parser.add_argument("--lr", default=1e-2, help="Learning rate of the Adam optimizer")
    parser.add_argument("--storeMI", default=True, help="If we are to only track the MI during training, and not save the activations")
    parser.add_argument("--allIdx", default=0, type=int, help="Whether to compute the MI plane with all data (1), or only training data (0)")
    parser.add_argument("--reduction", default=1, type=int, help="Reduce the amount of data processed")

    args = parser.parse_args()
    model_name, dataset_name = args.model, args.dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"you chose model: {model_name}, and dataset: {dataset_name}, epochs: {args.epochs}, with reduction {args.reduction}, model_size {args.hidden_dim}")
    print("learningRate:", args.lr)
    data, splits, model_params = get_data_and_indices(dataset_name, reduction_factor=args.reduction)

    model_params = {**model_params, "hidden_channels": args.hidden_dim, "activation": args.activation}

    model = get_and_create_model(model_name, model_parameters=model_params, device=device)
    print(model)
    print("args.allidx:", args.allIdx)
    if not args.allIdx:
        print("using training_idx")
        tracker = ActivationTracker(model, training_indices=splits["train"], store_MI=args.storeMI, y=data.y.unsqueeze(1))
    else:
        tracker = ActivationTracker(model, store_MI=args.storeMI, y=data.y.unsqueeze(1))

    losses, validation_accuracies, training_accuracies = train(
        model=model,
        tracker=tracker,
        split_indices=splits,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        criterion= torch.nn.CrossEntropyLoss(),
        data_=data,
        num_epochs=int(args.epochs),
        model_name=model_name,
        device=device
    )
    print("Finished training, writing data to file")
    now = time.strftime("%d%m%Y-%H%M%S")
    config = model_name + "-" + dataset_name + "-" + str(args.epochs) + "E-"
    PATH = "results/" + config + now
    
    if args.allIdx:
        PATH += "-AllIdx"
    
    PATH += "/"
    
    make_sure_path_exists(PATH)
    with open(PATH + "config.json", "w+") as f:
        json.dump(vars(args), f)
    with open(PATH + "data.pickle", "wb") as f:
        to_dump = {"losses": losses, "val_accs": validation_accuracies, "train_accs": training_accuracies}
        if args.storeMI:
            print("storing MI values to file")
            to_dump["storeMI"] = tracker.MI_STORE
        dill.dump(to_dump, file=f)

    print(f"Training finished, logged to directory {PATH}")