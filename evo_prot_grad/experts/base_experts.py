import torch 
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Any
from abc import ABC, abstractmethod
import evo_prot_grad.common.utils as utils
import evo_prot_grad.common.tokenizers as tokenizers


class Expert(ABC):
    """Defines a common interface for any type of expert. 
    """
    def __init__(
        self,
        temperature: float,
        model: nn.Module,
        vocab: Dict,
        device: str = "cpu",
        use_without_wildtype: bool = False
    ):
        """
        Args:
            temperature (float): Hyperparameter for re-scaling this expert in the Product of Experts.
            model (nn.Module): The model to use for the expert.
            vocab (Dict): The vocabulary for the expert.
            device (str): The device to use for the expert.
            use_without_wildtype (bool): Whether to use the expert without the wildtype,
                i.e., do not subtract the wildtype score from the expert score.
        """
        self.model = model
        self.temperature = temperature
        self.device = device
        self.use_without_wildtype = use_without_wildtype
        self.model.to(self.device)
        self.model.eval()
                
        # sort by vocab values
        self.alphabet = [k for k, v in sorted(vocab.items(), key=lambda item: item[1])]

        self.expert_to_canonical_order = utils.expert_alphabet_to_canonical(
                                         self.alphabet, self.device)
        
        # a torch scalar 
        self._wt_score = None
       
    @abstractmethod
    def _tokenize(self, inputs: List[str]) -> Any:
        """Tokenizes a list of protein sequences.

        Args:
            inputs (List[str]): A list of protein sequences.
        Returns:
            tokens (Any): tokenized sequence in whatever format the expert requires.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def _get_last_one_hots(self) -> torch.Tensor:
        """Returns the one-hot tensors *most recently passed* as input
        to this expert.

        The one-hot tensors are cached and accessed from 
        a evo_prot_grad.common.embeddings.OneHotEmbedding module, which
        we configure each expert to use.

        !!! warning
            This assumes that the desired one-hot tensors are the
            last tensors passed as input to the expert. If the expert
            is called twice, this will return the one-hot tensors from the
            second call. This is intended to address the issue that some experts take lists 
            of strings as input and internally converts them into one-hot tensors.
        """
        raise NotImplementedError()
    

    @abstractmethod
    def _model_output_to_scalar_score(self,
                                       model_output: torch.Tensor,
                                       **kwargs) -> torch.Tensor:
        """Converts the model output to a scalar score. 

        Args:
            model_output (torch.Tensor): The output of the expert model.
            **kwargs (Dict): Any additional arguments required by the expert.
        Returns:
            score (torch.Tensor): The scalar score.
        """
        raise NotImplementedError()
    
    ####### "Public" methods #######

    @abstractmethod
    def set_wt_score(self, wt_seq: str) -> None:
        """Sets the wildtype score value for protein wt_seq.

        Args:
            wt_seq (str): The wildtype sequence.
        
        Returns:
            None
        """
        raise NotImplementedError()
        
    @abstractmethod
    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the expert score for a batch of protein sequences as well as 
           the one-hot encoded input sequences for which a gradient can be computed.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            expert_score (torch.Tensor): of shape [parallel_chains]
        """
        raise NotImplementedError()

 
class HuggingFaceExpert(Expert):
    def __init__(self,
                 temperature: float, 
                 model: nn.Module,
                 vocab: Dict,
                 device: str,
                 use_without_wildtype: bool):
        """
        Args:
            temperature (float): Hyperparameter for re-scaling this expert in the Product of Experts.
            model (nn.Module): The model to use for the expert.
            vocab (Dict): The vocab to use for the expert.
            device (str): The device to use for the expert.
            use_without_wildtype (bool): Whether to use the expert without the wildtype.
        """
        super().__init__(temperature, model, vocab, device, use_without_wildtype)


    def set_wt_score(self, wt_seq: str) -> None:
        """ Sets the score value for wildtype protein wt_seq.

        Args:
            wt_seq (str): The wildtype sequence.
        """
        encoded_inputs = self._tokenize([wt_seq])
        # Always call get_last_one_hots() right after
        # calling self.model()
        logits = self.model(**encoded_inputs).logits
        oh = self._get_last_one_hots()

        self._wt_score = self._model_output_to_scalar_score(oh, logits=logits).detach()


    def _model_output_to_scalar_score(self, x_oh: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Returns the scalar score assuming the expert predicts
        a logit score for each amino acid.

        Args:
            x_oh: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
            logits: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
        Returns: 
            score (torch.Tensor): of shape [parallel_chains]
        """
        return (x_oh * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=[1,2])    


    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the one-hot sequences and expert score.
        Assumes the PLM model predicts a logit score for each amino acid.
        
        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            expert_score (torch.Tensor): of shape [parallel_chains]
        """
        if not self.use_without_wildtype:
            assert self._wt_score is not None, \
                "Wildtype score must be set before calling the expert."

        encoded_inputs = self._tokenize(inputs)
        # All HF PLMs output a ModelOutput object with a logits attribute
        logits = self.model(**encoded_inputs).logits
        oh = self._get_last_one_hots()
        if self.use_without_wildtype:
            score = self._model_output_to_scalar_score(oh, logits=logits)
        else:
            score = self._model_output_to_scalar_score(oh, logits=logits) - self._wt_score
        return oh, score 
    

class AttributeExpert(Expert):
    """Interface for experts trained (typically with supervised learning)
    to predict an attribute (e.g., activity or stability) from one-hot encoded sequences.
    """
    def __init__(self, 
                 temperature: float,
                 model: nn.Module,
                 device: str,
                 use_without_wildtype: bool,
                 tokenizer: Optional[tokenizers.ExpertTokenizer] = None):
        """
        Args:
            temperature (float): Hyperparameter for re-scaling this expert in the Product of Experts.
            model (nn.Module): The model to use for the expert.
            tokenizer (ExpertTokenizer): The tokenizer to use for the expert.
            use_without_wildtype (bool): Whether to use the expert without the wildtype.
            device (str): The device to use for the expert.
        """
        if tokenizer is None:
            tokenizer = tokenizers.OneHotTokenizer(utils.CANONICAL_ALPHABET)
        super().__init__(temperature, model, tokenizer.get_vocab(), device, use_without_wildtype)
        self.tokenizer = tokenizer

    def set_wt_score(self, wt_seq: str) -> None:
        """Sets the score value for wildtype protein wt_seq.
        
        Args:
            wt_seq (str): The wildtype sequence.
        """
        encoded_inputs = self._tokenize([wt_seq])
        y = self.model(encoded_inputs)
        self.wt_score = self._model_output_to_scalar_score(y)

    def _model_output_to_scalar_score(self, model_outputs: torch.Tensor) -> torch.Tensor:
        """Returns the score for the given input assuming 
        the expert predicts a single scalar. 

        Args:
            model_outputs: (torch.Tensor) of shape [parallel_chains]
        Returns:
            model_outputs: (torch.Tensor) of shape [parallel_chains]
        """
        assert model_outputs.dim() == 1, "Model output must be a scalar."
        return model_outputs
    
    def _get_last_one_hots(self) -> torch.Tensor:
        """Unused."""
        raise NotImplementedError()
    
    def _tokenize(self, inputs: List[str]):
        """ Tokenizes a list of protein sequences.
        
        Args:
            inputs (List[str]): A list of protein sequences.
        """
        return self.tokenizer(inputs).to(self.device)
        
    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            score (torch.Tensor): of shape [parallel_chains]
        """
        if not self.use_without_wildtype:
            assert self.wt_score is not None, \
                "Wildtype score must be set before calling the expert."
        encoded_oh_inputs = self._tokenize(inputs)
        encoded_oh_inputs = encoded_oh_inputs.requires_grad_()
        y = self.model(encoded_oh_inputs)
        if self.use_without_wildtype:
            score = self._model_output_to_scalar_score(y)
        else:
            score = self._model_output_to_scalar_score(y) - self.wt_score
        return encoded_oh_inputs, score 