from typing import overload, Literal

import torch

from gerbilizer.outputs import ModelOutput, ModelOutputFactory

class GerbilizerArchitecture(torch.nn.Module):

    defaults: dict

    def __init__(self, CONFIG, output_factory: ModelOutputFactory):
        super(GerbilizerArchitecture, self).__init__()

        self.config = CONFIG
        self.output_factory = output_factory
        self.n_outputs = self.output_factory.n_outputs_expected

    def load_weights(self, best_weights_path = None, use_final_weights: bool = False):
        """
        Load in the weights of the model, where use of `best_weights.pt` or
        `final_weights.pt` is specified by `use_final_weights`.
        """
        if best_weights_path is None:
            # try checking config for a weights path
            print('No weights path provided, checking model config.')
            try:
                best_weights_path = self.config['WEIGHTS_PATH']
            except KeyError:
                raise ValueError(
                    'Couldn\'t load model weights! No `best_weights_path` kwarg provided '
                    'to `load_weights`, and could not find "WEIGHTS_PATH" in model config.'
                    )

        weights_path = best_weights_path
        if use_final_weights:
            weights_path = weights_path.replace('best', 'final')

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(device)

        weights = torch.load(weights_path, map_location=device)
        self.load_state_dict(weights)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    # add overload for nice unbatched functionality
    @overload
    def forward(self, x: torch.Tensor, unbatched: Literal[False]) -> ModelOutput: ...
    @overload
    def forward(self, x: torch.Tensor, unbatched: Literal[True]) -> list[ModelOutput]: ...

    def forward(self, x: torch.Tensor, unbatched: bool = False):
        """
        Run the model on input `x` and return an appropriate
        ModelOutput object, determined based on `self.output_factory`.
        """
        raw_outputs = self._forward(x)
        if unbatched:
            outputs = []
            for o in raw_outputs:
                # add batch dim back
                outputs.append(self.output_factory.create_output(o[None]))
            return outputs
        else:
            return self.output_factory.create_output(self._forward(x))
