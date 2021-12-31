import numpy as np
from estimators import KOLCHINSKY_MUTUAL_INFO_COMPUTATION


class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


class ActivationTracker:
    """
    This is a general method for tracking activations of any PyTorch network the user defines,
    and we rely on the named_children attribute of the PyTorch model
    """
    def __init__(self, module, store_MI, training_indices=None, y=None):
        """
        if store_MI: We do not store/track the activations for each layer and each epoch, but we rather directly store
                     the computed MI-values (currently only using the method of pairwise distances from https://arxiv.org/abs/1706.02419)
        """
        self.forward_hooks = {name:Hook(layer) for name, layer in list(module.named_children())}
        self.epoch_activations = []
        self.epoch_number = -1
        self.new_epoch = False
        self.training_indices = training_indices
        self.y = y
        self.MI_STORE = [] if store_MI else None

    def register_new_epoch(self, what_to_save):
        if self.MI_STORE is not None:
            self.MI_STORE.append({key: None for key in self.forward_hooks.keys() if key in what_to_save})
        self.epoch_activations.append({key: None for key in self.forward_hooks.keys() if key in what_to_save})
        self.epoch_number += 1

    def __len__(self):
        return len(self.epoch_activations)  # same as num_epochs

    def save(self):
        # y here should be fore the whole dataset, i.e. same that was passed to model.forward()
        for key in self.epoch_activations[self.epoch_number].keys():
            if self.training_indices is not None:
                activations = self.forward_hooks[key].output.detach().cpu()[self.training_indices].numpy()
            else:
                activations = self.forward_hooks[key].output.detach().cpu().numpy()

            if self.MI_STORE is not None and self.y is not None:
                if self.training_indices is not None:
                    y = self.y[self.training_indices]
                else:
                    y = self.y
                IXT, ITY = KOLCHINSKY_MUTUAL_INFO_COMPUTATION(activations, y)
                self.MI_STORE[self.epoch_number][key] = [IXT, ITY]
            else:
                if self.epoch_activations[self.epoch_number][key] is None:
                    self.epoch_activations[self.epoch_number][key] =  activations
                else:
                    # if .save() called on each batch
                    self.epoch_activations[self.epoch_number][key] = np.concatenate([self.epoch_activations[self.epoch_number][key],
                                                                            activations], axis=0)
