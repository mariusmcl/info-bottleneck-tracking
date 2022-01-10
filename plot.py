import argparse
import dill
import numpy as np
from plotting_utils import plot_train_and_val_accuracies, plot_information_plane, plot_information_plane_paper


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Where the storage files are locatde")
    parser.add_argument("--directory", default="results/GCN-Cora-30122021-230616")
    parser.add_argument("--everyN", default=10, type=int, help="At what interval to plot the MI values")
    parser.add_argument("--inset", default="", type=str, help="Whether to plot with a custom inset")

    args = parser.parse_args()
    
    print("Directory:", args.directory, "everyN:", args.everyN)
    with open(args.directory + "/data.pickle", "rb") as f:
        data = dill.load(f)
    
    # Assume we have computed the MI-values during training:

    MI_values = data["storeMI"]  #  [{act1: [x, y], act2: [x, y]}, {act1: [x, y], act2: [x, y]}, .... ] 

    num_epochs, num_representations = len(MI_values), len(MI_values[0].keys())
    IXT_array = np.zeros(shape=(num_epochs, num_representations))
    ITY_array = np.zeros(shape=(num_epochs, num_representations))

    for epoch, info_dict in enumerate(MI_values):
        for i in range(1, num_representations + 1):
            IXT, ITY = info_dict["act" + str(i)]
            IXT_array[epoch, i-1], ITY_array[epoch, i-1] = IXT, ITY

    PATH = args.directory + "/"
    val_accs, train_accs = data["val_accs"], data["train_accs"]
    items = PATH.split("-")
    model_name, dataset_name = items[0], items[1]

    #plot_train_and_val_accuracies(train_accs=train_accs,val_accs=val_accs,PATH=PATH,model_name=model_name,dataset_name=dataset_name)

    if len(args.inset) > 0:
        print("Creating inset:")
        plot_information_plane_paper(
            IXT_array=IXT_array,
            ITY_array=ITY_array,
            num_epochs=num_epochs,
            every_n=1,
            PATH=PATH,
            inset=args.inset
        )
    else:
        plot_information_plane(
            IXT_array=IXT_array,
            ITY_array=ITY_array,
            num_epochs=num_epochs,
            PATH=PATH,
            every_n=1,
            model_name=model_name,
            dataset_name=dataset_name
        )