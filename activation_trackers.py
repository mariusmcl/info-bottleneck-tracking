import numpy as np


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
    def __init__(self, module, backward=False, store_snr=True, training_indices=None):
        self.forward_hooks = {name:Hook(layer) for name, layer in list(module.named_children())}
        self.epoch_activations = []
        self.epoch_number = -1
        self.new_epoch = False
        self.module = module     # have a reference to this here in order to be able to store
        self.training_indices = training_indices

    def register_new_epoch(self, what_to_save, backward_save=None):
        # Register all last stored activations (since previous register_new_epoch) into a single epoch
        self.epoch_activations.append({key: None for key in self.forward_hooks.keys() if key in what_to_save})
        self.epoch_number += 1

    def __len__(self):
        return len(self.epoch_activations)  # same as num_epochs

    def save(self):
        # for each batch in current epoch, append the latest batch's activations to that epoch's dictionary
        for key in self.epoch_activations[self.epoch_number].keys():
                        
            if self.training_indices is not None:
                activations = self.forward_hooks[key].output.detach().cpu()[self.training_indices].numpy()
            else:
                activations = self.forward_hooks[key].output.detach().cpu().numpy()
            
            if self.epoch_activations[self.epoch_number][key] is None:
                self.epoch_activations[self.epoch_number][key] =  activations  #.astype(np.float16)
            else:
                self.epoch_activations[self.epoch_number][key] = np.concatenate([self.epoch_activations[self.epoch_number][key],
                                                                        activations], axis=0)
