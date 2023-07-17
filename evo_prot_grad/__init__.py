import importlib
from typing import Optional, Union
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from evo_prot_grad.experts.base_experts import Expert
from evo_prot_grad.common.tokenizers import ExpertTokenizer
from evo_prot_grad.common.sampler import DirectedEvolution

def get_expert(expert_name: str,
               temperature: float = 1.0,               
               model: Optional[nn.Module] = None,
               tokenizer: Optional[Union[ExpertTokenizer, PreTrainedTokenizerBase]] = None,
               device: str = 'cpu',
               use_without_wildtype: bool = False) -> Expert:
    """

    Args:
        expert_name (str): Name of the expert to be used.
        temperature (float, optional): Temperature for the expert. Defaults to 1.0.
        model (Optional[nn.Module], optional): Model to be used for the expert. Defaults to None.
        tokenizer (Optional[Union[ExpertTokenizer, PreTrainedTokenizerBase]], optional): Tokenizer to be used for the expert. Defaults to None.
        device (str, optional): Device to be used for the expert. Defaults to 'cpu'.
    
    Raises:
        ValueError: If the expert name is not found.

    Returns:
        expert (Expert): An instance of the expert.
    """
    try:
        expert_mod = importlib.import_module(f"evo_prot_grad.experts.{expert_name}_expert")
        return expert_mod.build(
            temperature = temperature,
            model = model,
            tokenizer = tokenizer,
            device = device,
            use_without_wildtype = use_without_wildtype
        )
    except:
        raise ValueError(f"Expert {expert_name} not found in evo_prot_grad.experts.")
