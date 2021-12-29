from numpy import mod
import torch
import argparse
from activation_trackers import ActivationTracker
from models import get_and_create_model
from data_pipelines import get_data_and_indices



def train(model, tracker, optimizer, criterion, data_, split_indices, num_epochs, verbose=True):
    losses, val_accuracies, train_accuracies = [], [], []
    for epoch in range(1, num_epochs):
        tracker.register_new_epoch(['act1', 'act2', 'act3'])
        model.train()
        optimizer.zero_grad()
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
        
        #if verbose:
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Valacc{val_acc:.4f}, Trainacc {acc:.4f}')
    return losses, val_accuracies, train_accuracies


if __name__ == "__main__":

    model_choices = set(("GCN", "GAT", "GIN", "MLP"))
    dataset_choices = set(("Cora", "arxiv", "products"))

    parser = argparse.ArgumentParser(description="Trains a chosen model on a dataset")
    parser.add_argument("--model", default="GCN", choices=model_choices)
    parser.add_argument("--dataset", default="Cora", choices=dataset_choices)
    parser.add_argument("--hidden_dim", default=200, help="Number of neurons in the hidden layers")
    parser.add_argument("--activation", default="relu", help="Activation to be used for hidden layers in the network")
    parser.add_argument("--epochs", default=100, help="Number of epochs to train with gradient descent. Note that we use the entire dataset for each gradient update")
    parser.add_argument("--lr", default=1e-3, help="Learning rate of the Adam optimizer")

    args = parser.parse_args()
    model_name, dataset_name = args.model, args.dataset

    data, splits, model_params = get_data_and_indices(dataset_name)

    model_params = {**model_params, "hidden_channels": args.hidden_dim, "activation": args.activation}

    model = get_and_create_model(model_name, model_parameters=model_params)

    tracker = ActivationTracker(model, training_indices=splits["train"])

    losses, validation_accuracies, training_accuracies = train(
        model=model,
        tracker=tracker,
        split_indices=splits,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        criterion= torch.nn.CrossEntropyLoss(),
        data_=data,
        num_epochs=args.epochs
    )

    #print(model, dataset)
    print(f"you chose model: {model_name}, and dataset: {dataset_name}")

    a=2


