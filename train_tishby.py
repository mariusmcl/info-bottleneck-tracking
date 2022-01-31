from os import devnull
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



def train_tishby(model, tracker, optimizer, criterion, X, y, split_indices, num_epochs, verbose=True):
    losses, val_accuracies, train_accuracies = [], [], []

    for epoch in range(1, num_epochs):
        tracker.register_new_epoch(['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        model.train()
        optimizer.zero_grad()

        out = model(X)
        train_loss = criterion(out[split_indices["train"]], y[split_indices["train"]].float())

        train_loss.backward()
        tracker.save()
        losses.append(train_loss.cpu().item())
        optimizer.step()
        with torch.no_grad():
            pred = (out > 0.5).long()
            acc = (pred[split_indices["train"]] ==  y[split_indices["train"]]).sum() / pred[split_indices["train"]].shape[0]
            train_accuracies.append(acc)

            val_acc = (pred[split_indices["valid"]] == y[split_indices["valid"]]).sum() / pred[split_indices["valid"]].shape[0]
            val_accuracies.append(val_acc)
        
        if verbose:
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Valacc{val_acc:.4f}, Trainacc {acc:.4f}')

    return losses, val_accuracies, train_accuracies


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trains a chosen model on a dataset")
    parser.add_argument("--activation", default="relu", help="Activation to be used for hidden layers in the network")
    parser.add_argument("--epochs", default=100, help="Number of epochs to train with gradient descent. Note that we use the entire dataset for each gradient update")
    parser.add_argument("--lr", default=1e-2, help="Learning rate of the Adam optimizer")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"you chose model to train TishbyNet on TishbyData! epochs: {args.epochs}, model_size 'Default'")
    print("learningRate:", args.lr)

    X, y, splits = get_data_and_indices("Tishby")

    model = get_and_create_model("TishbyNet", device=device)
    

    tracker = ActivationTracker(model.net, training_indices=splits["train"], y=y, store_MI=True)

    losses, validation_accuracies, training_accuracies = train_tishby(
        model=model,
        tracker=tracker,
        split_indices=splits,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        criterion= torch.nn.BCEWithLogitsLoss(),
        X=X,
        y=y,
        num_epochs=int(args.epochs),
    )
    print("Finished training, writing data to file")
    now = time.strftime("%d%m%Y-%H%M%S")
    config = "TISHBY" + "-"  + str(args.epochs) + "E-"
    PATH = "results/" + config + now
    
    PATH += "/"
    
    make_sure_path_exists(PATH)
    with open(PATH + "config.json", "w+") as f:
        json.dump(vars(args), f)
    with open(PATH + "data.pickle", "wb") as f:
        to_dump = {"losses": losses, "val_accs": validation_accuracies, "train_accs": training_accuracies}
        print("storing MI values to file")
        to_dump["storeMI"] = tracker.MI_STORE
        dill.dump(to_dump, file=f)

    print(f"Training finished, logged to directory {PATH}")