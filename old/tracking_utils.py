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



class DataTracker:
    # AS OF NOW: så lagres alle de gradientene over hele epoken. dersom dette blir infeasible minnemessig
    # må det vurderes å estimere stdev/mean
    def __init__(self, module, backward=False, store_snr=True):
        self.forward_hooks = {name:Hook(layer) for name, layer in list(module.named_children())}
        self.backward = backward
        if backward:
            self.backward_hooks = {name:Hook(layer, backward=True) for name, layer in list(module.named_children())}
            self.gradients = []
        else:
            self.backward_hooks = None
        self.epoch_activations = []
        self.store_snr = store_snr
        self.epoch_number = -1
        self.new_epoch = False
        self.module = module     # have a reference to this here in order to be able to store
        self.grad_mean_snr_epochs = []   # add the grad mean SNR's here
        self.grad_std_snr_epochs = []   # add the std SNR's here   --> both are of len(epochs) @ end of training
        self.weight_norms = []
    def register_new_epoch(self, what_to_save, backward_save=None):
        # Register all last stored activations (since previous register_new_epoch) into a single epoch
        self.epoch_activations.append({key: None for key in self.forward_hooks.keys() if key in what_to_save})
        if self.backward:
            self.gradients.append({key: None for key in self.backward_hooks.keys() if key in backward_save})
        if self.backward and self.store_snr and len(self.epoch_activations) > 1:  # we are storing from the previous epoch, so first time @ 2nd epoch
            # Now we will be performing a mean and stdev over (n_train // batch_size) elements, where each element is a grad for a layer weight
            layer_snr_mean_tracker = {layer_name: 0 for layer_name in backward_save}
            layer_snr_stdev_tracker = {layer_name: 0 for layer_name in backward_save}
            for key in self.gradients[self.epoch_number].keys():
                epoch_mean_gradient = np.mean(self.gradients[self.epoch_number][key], axis=0)

                weight, bias = self.module._modules[key].weight.data.cpu().numpy(), self.module._modules[key].bias.data.cpu().numpy()
                full_weight = np.concatenate([weight, np.expand_dims(bias, axis=1)], axis=1)
                norm_constant = np.linalg.norm(full_weight, 2)    # AKKURATT NÅ!! så tar vi bare  vekten på slutten av epoken, VEKTEN KAN HA ENDRET SEG MYE ILA. EPOKEN!!!
                #norm_constant = norm_constant * full_weight.shape[1]
                #print(norm_constant)
                #norm_constant = 1
                #val = np.linalg.norm(epoch_mean_gradient)
                layer_snr_mean_tracker[key] = np.linalg.norm(epoch_mean_gradient) / norm_constant    # now same size as the key's weight matrix
                #check = [grad for grad in self.gradients[self.epoch_number][key]]
                epoch_stdev_gradient = np.std(self.gradients[self.epoch_number][key], axis=0)
                layer_snr_stdev_tracker[key] = np.linalg.norm(epoch_stdev_gradient) / norm_constant
                self.weight_norms.append(norm_constant)
            self.grad_mean_snr_epochs.append(layer_snr_mean_tracker)
            self.grad_std_snr_epochs.append(layer_snr_stdev_tracker)
        self.epoch_number += 1

    def __len__(self):
        return len(self.epoch_activations)  # sammenfaller med num_epochs

    def save(self):
        # for each batch in current epoch, append the latest batch's activations to that epoch's dictionary
        for key in self.epoch_activations[self.epoch_number].keys():
            if self.epoch_activations[self.epoch_number][key] is None:
                self.epoch_activations[self.epoch_number][key] = self.forward_hooks[key].output.detach().cpu().numpy()#.astype(np.float16)
            else:
                self.epoch_activations[self.epoch_number][key] = np.concatenate([self.epoch_activations[self.epoch_number][key],
                                                                        self.forward_hooks[key].output.detach().cpu().numpy()], axis=0)
        if self.backward:
            for key in self.gradients[self.epoch_number].keys():
                if self.gradients[self.epoch_number][key] is None:
                    self.gradients[self.epoch_number][key] = np.expand_dims(self.backward_hooks[key].input[2].detach().cpu().numpy(), axis=0)
                else:
                    self.gradients[self.epoch_number][key] = np.concatenate([self.gradients[self.epoch_number][key],
                                                                             np.expand_dims(self.backward_hooks[key].input[2].detach().cpu().numpy(), axis=0)], axis=0)




class GCNDataTracker:
    # AS OF NOW: så lagres alle de gradientene over hele epoken. dersom dette blir infeasible minnemessig
    # må det vurderes å estimere stdev/mean
    def __init__(self, module, backward=False, store_snr=True):
        self.forward_hooks = {name:Hook(layer) for name, layer in list(module.named_children())}
        self.backward = backward
        if backward:
            self.backward_hooks = {name:Hook(layer, backward=True) for name, layer in list(module.named_children())}
            self.gradients = []
        else:
            self.backward_hooks = None
        self.epoch_activations = []
        self.store_snr = store_snr
        self.epoch_number = -1
        self.new_epoch = False
        self.module = module     # have a reference to this here in order to be able to store
        self.grad_mean_snr_epochs = []   # add the grad mean SNR's here
        self.grad_std_snr_epochs = []   # add the std SNR's here   --> both are of len(epochs) @ end of training
        self.weight_norms = []
    def register_new_epoch(self, what_to_save, backward_save=None):
        # Register all last stored activations (since previous register_new_epoch) into a single epoch
        self.epoch_activations.append({key: None for key in self.forward_hooks.keys() if key in what_to_save})
        if self.backward:
            self.gradients.append({key: None for key in self.backward_hooks.keys() if key in backward_save})
        if self.backward and self.store_snr and len(self.epoch_activations) > 1:  # we are storing from the previous epoch, so first time @ 2nd epoch
            # Now we will be performing a mean and stdev over (n_train // batch_size) elements, where each element is a grad for a layer weight
            layer_snr_mean_tracker = {layer_name: 0 for layer_name in backward_save}
            layer_snr_stdev_tracker = {layer_name: 0 for layer_name in backward_save}
            for key in self.gradients[self.epoch_number].keys():
                epoch_mean_gradient = np.mean(self.gradients[self.epoch_number][key], axis=0)

                weight, bias = self.module._modules[key].weight.data.cpu().numpy(), self.module._modules[key].bias.data.cpu().numpy()
                tmp = np.expand_dims(bias, axis=0)
                full_weight = np.concatenate([weight, tmp], axis=0)   # NB ENDRET CONCATENATE TIL AXIS=0 HER!!
                norm_constant = np.linalg.norm(full_weight, 2)    # AKKURATT NÅ!! så tar vi bare  vekten på slutten av epoken, VEKTEN KAN HA ENDRET SEG MYE ILA. EPOKEN!!!
                #norm_constant = norm_constant * full_weight.shape[1]
                #print(norm_constant)
                #norm_constant = 1
                #val = np.linalg.norm(epoch_mean_gradient)
                layer_snr_mean_tracker[key] = np.linalg.norm(epoch_mean_gradient) / norm_constant    # now same size as the key's weight matrix
                #check = [grad for grad in self.gradients[self.epoch_number][key]]
                epoch_stdev_gradient = np.std(self.gradients[self.epoch_number][key], axis=0)
                layer_snr_stdev_tracker[key] = np.linalg.norm(epoch_stdev_gradient) / norm_constant
                self.weight_norms.append(norm_constant)
            self.grad_mean_snr_epochs.append(layer_snr_mean_tracker)
            self.grad_std_snr_epochs.append(layer_snr_stdev_tracker)
        self.epoch_number += 1

    def __len__(self):
        return len(self.epoch_activations)  # sammenfaller med num_epochs

    def save(self):
        # for each batch in current epoch, append the latest batch's activations to that epoch's dictionary
        for key in self.epoch_activations[self.epoch_number].keys():
            if self.epoch_activations[self.epoch_number][key] is None:
                self.epoch_activations[self.epoch_number][key] = self.forward_hooks[key].output.detach().cpu().numpy()#.astype(np.float16)
            else:
                self.epoch_activations[self.epoch_number][key] = np.concatenate([self.epoch_activations[self.epoch_number][key],
                                                                        self.forward_hooks[key].output.detach().cpu().numpy()], axis=0)
        if self.backward:
            for key in self.gradients[self.epoch_number].keys():
                idx = 0    # FOR THIS CUSTOM GCN CLASS!!!!
                #if
                if self.gradients[self.epoch_number][key] is None:
                    # når grafen clustres kan det være forskjellige # noder i hver cluster, som gir problemer når man skal ta mean
                    # osv. tok :600 nå siden alle clusters > 600
                    self.gradients[self.epoch_number][key] = np.expand_dims(self.backward_hooks[key].input[0].detach().cpu().numpy(), axis=0)
                else:
                    self.gradients[self.epoch_number][key] = np.concatenate([self.gradients[self.epoch_number][key],
                                                                             np.expand_dims(self.backward_hooks[key].input[0].detach().cpu().numpy(), axis=0)], axis=0)
