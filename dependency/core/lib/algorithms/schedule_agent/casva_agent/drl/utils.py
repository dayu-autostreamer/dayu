import torch


def format_input_state(state: torch.Tensor):
    if state.dim() == 2:
        state = state.unsqueeze(0)
    return state
