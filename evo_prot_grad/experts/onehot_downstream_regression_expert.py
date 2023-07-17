from evo_prot_grad.experts.base_experts import AttributeExpert
from evo_prot_grad.common.tokenizers import OneHotTokenizer
import evo_prot_grad.common.utils as utils
from torch.nn import Module
from typing import Optional


class OneHotDownstreamRegressionExpert(AttributeExpert):
    """ Basic one-hot regression expert."""
    def __init__(self, 
                 temperature: float,
                 model: Module,
                 device: str,
                 tokenizer: Optional[OneHotTokenizer] = None,
                 use_without_wildtype: bool = False):
        """
        Args:
            temperature (float): Temperature for sampling from the expert.
            model (Module): The model to use for the expert.
            device (str): The device to use for the expert.
            tokenizer (Optional[OneHotTokenizer], optional): The tokenizer to use for the expert. If None,
                a OneHotTokenizer will be constructed. Defaults to None.
            use_without_wildtype (bool): Whether to use the expert without the wildtype. Defaults to False.
        """
        if tokenizer is None:
            tokenizer = OneHotTokenizer(utils.CANONICAL_ALPHABET)
        super().__init__(temperature,
                        model,
                        device,
                        use_without_wildtype,
                        tokenizer)
        
def build(**kwargs):
    """Builds a OneHotDownstreamExpert."""
    return OneHotDownstreamRegressionExpert(**kwargs)