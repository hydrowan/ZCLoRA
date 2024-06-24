LoRA class that utilises zero initialised convolutions, this means that the LoRA does not degrade the model at all initially, by learning the weight updates in a residual manner.
This idea was built to fine tune LLMs with very limited data, when catastrophic interference was not an issue, for a project involving continuous fine tuning of an instruct model.

Note that the loss landscape is not as simple as standard LoRA; the ControlNet authors introduced the idea of using ZC in their paper (to gate skip connections) and discuss the atypical sudden improvement found in training as the ZC start to flip and apply.

This class is designed to replace Linear layers in Mistral, may require modification for specific use-cases.
