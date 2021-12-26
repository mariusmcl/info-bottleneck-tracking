"Old" code, which also contains functionality for tracking the gradient for different PyTorch models.

Tracking the gradient of different PyTorch models is slightly more involved, as one has to make sure one gets the right
"Index" from the model, i.e.

self.backward_hooks[key].input[2] for an MLP (in the class DataTracker)

self.backward_hooks[key].input[0] for a GCN (in the class GCNDataTracker)

If one combines different models (i.e. GCN, MLP and RNN) in one PyTorch model, one would have to augment the functionality in order to track
different keys depending on the layer (not that much more work)
