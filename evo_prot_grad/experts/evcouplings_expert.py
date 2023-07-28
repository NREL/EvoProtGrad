from evo_prot_grad.experts.base_experts import Expert
from evo_prot_grad.common.tokenizers import OneHotTokenizer
import evo_prot_grad.common.utils as utils
import evo_prot_grad.models.potts as potts
from typing import List, Tuple, Optional
import torch


class EVCouplingsExpert(Expert):
    """Expert class for EVCouplings Potts models.
    EVCouplings lib uses the canonical alphabet by default.
    """
    def __init__(self, 
                 temperature: float,
                 model: potts.EVCouplings,
                 device: str,
                 tokenizer: Optional[OneHotTokenizer] = None,
                 use_without_wildtype: bool = False):
        """
        Args:
            temperature (float): Temperature for sampling from the expert.
            model (potts.EVCouplings): The model to use for the expert.
            device (str): The device to use for the expert.
            tokenizer (Optional[OneHotTokenizer]): The tokenizer to use for the expert. If None, uses
                    OneHotTokenizer(utils.CANONICAL_ALPHABET, device).
            use_without_wildtype (bool): Whether to use the expert without the wildtype.
        """
        assert model is not None, "EVCouplingsExpert requires a potts.EVCouplings model to be provided."
        if tokenizer is None:
            tokenizer = OneHotTokenizer(utils.CANONICAL_ALPHABET)
        super().__init__(temperature,
                         model, 
                         tokenizer.get_vocab(),
                         device=device,
                         use_without_wildtype=use_without_wildtype)
        assert model.alphabet == self.alphabet, \
            f"EVcouplings alphabet {model.alphabet} should match our canonical alphabet {self.alphabet}"
        self.tokenizer = tokenizer
        
    def set_wt_score(self, wt_seq: str) -> None:
        """Sets the wildtype score value for protein wt_seq
        """
        encoded_inputs = self._tokenize([wt_seq])
        self._wt_score = self.model(encoded_inputs)
        

    def _get_last_one_hots(self) -> torch.Tensor:
        return self.model.one_hot_embedding.one_hots
    
    def _model_output_to_scalar_score(self, hamiltonian: torch.Tensor):
        """ Hamiltonian  """
        return hamiltonian

    def _tokenize(self, inputs: List[str]) -> torch.FloatTensor:
        return self.tokenizer(inputs).to(self.device)
    
    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            expert_score (torch.Tensor): of shape [parallel_chains]
        """
        if not self.use_without_wildtype:
            assert self._wt_score is not None, "Wildtype score must be set before calling the expert."

        encoded_inputs = self._tokenize(inputs)
        hamiltonian = self.model(encoded_inputs)
        oh = self._get_last_one_hots()
        if self.use_without_wildtype:
            score = self._model_output_to_scalar_score(hamiltonian)
        else:
            score = self._model_output_to_scalar_score(hamiltonian) - self._wt_score
        return oh, score 
    
def build(**kwargs):
    return EVCouplingsExpert(**kwargs)