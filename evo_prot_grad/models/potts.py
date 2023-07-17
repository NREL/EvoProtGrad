import torch
import torch.nn as nn
from pathlib import Path
from evcouplings.couplings import CouplingsModel
import evo_prot_grad.common.embeddings as embeddings


class EVCouplings(nn.Module):
    """EVCoupling Potts model implemented in PyTorch.

    Represents a Potts model with a single coupling matrix and a single bias vector
    for a specific region (i.e., subsequence) of the wild type protein sequence
    under directed evolution.  
    """
    def __init__(self, model_params_file: str, fasta_file: str):
        super().__init__()
        """
        Args:
            model_params_file (str): Path to the model parameters file in plmc_v2 format.
            fasta_file (str): Path to the FASTA file containing the wild-type sequence.
        """
        self.one_hot_embedding = embeddings.IdentityEmbedding()

        c = CouplingsModel(model_params_file)

        self.alphabet = [k for k,v in sorted(c.alphabet_map.items(), key=lambda item: item[1])]
        # the subsequence of the wild-type sequence covered by the Potts model.
        # we will use this to index into the one-hot tensors.
        self.index_list = c.index_list
        # Adjust index_list to account for the fact that
        # the subsequence covered by the Potts model (which is 
        # based on the MSA) may not start
        # at index 0 of the wild-type sequence, and that the
        # wild type sequence may be a subsequence of the protein.
        with open(Path(fasta_file), 'r') as f:
            for line in f:
                if line[0] == '>': # comment
                    comment = line.strip()                      
                    if '/' in comment:
                        wild_type_start = int(comment.split('/')[-1].split('-')[0])
                    else:
                        wild_type_start = 1
                    assert self.index_list[0] >= wild_type_start, \
                        f"wild_type_start: {wild_type_start}, index_list[0]: {self.index_list[0]}"
                    self.index_list -= wild_type_start
                    break

        self.J = nn.Parameter(
            torch.from_numpy(c.J_ij).float(), requires_grad=True)
        self.L, _, self.V, _ = self.J.shape

        self.h = nn.Parameter(
            torch.from_numpy(c.h_i).float(), requires_grad=True)


    def _hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): one-hot tensor of shape [parallel_chains, seq_len, vocab_size]

        Returns:
            hamiltonian (torch.Tensor): shape [parallel_chains]
        """
        Jx = torch.einsum("ijkl,bjl->bik", self.J, x)
        xJx = torch.einsum("aik,aik->a", Jx, x)  / 2  # J_ij == J_ji. J_ii is zero.
        bias = (self.h[None] * x).sum(-1).sum(-1)
        return xJx + bias
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): one-hot tensor of shape [parallel_chains, seq_len, vocab_size]
        
        Returns:
            hamiltonian (torch.Tensor): shape [parallel_chains]
        """
        x = self.one_hot_embedding(x)
        x = x[:,self.index_list[0]:self.index_list[-1]+1]

        return self._hamiltonian(x) 
        

