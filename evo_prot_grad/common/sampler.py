from typing import List, Tuple, Optional
import torch
import numpy as np
from pathlib import Path
from evo_prot_grad.experts.base_experts import Expert
import evo_prot_grad.common.utils as utils
import evo_prot_grad.common.tokenizers as tokenizers


class DirectedEvolution:
    """Main class for plug and play directed evolution with gradient-based discrete MCMC.
    """
    def __init__(self, 
                 experts: List[Expert],
                 parallel_chains: int,
                 n_steps: int,
                 max_mutations: int,
                 output: str = 'last',
                 preserved_regions: Optional[List[Tuple[int, int]]] = None,
                 wt_protein: Optional[str] = None,
                 wt_fasta: Optional[str] = None,
                 verbose: bool = False,
                 random_seed: Optional[int] = None):
        """
        Args:
            experts (List[Expert]): List of experts
            parallel_chains (int): number of parallel chains
            n_steps (int): number of steps to run directed evolution
            max_mutations (int): maximum mutation distance from WT, disable by setting to -1.
            output (str): output type, either 'best', 'last' or 'all'. Default is 'last'.
            preserved_regions (List[Tuple[int,int]]): list of tuples of (start, end) of preserved regions. Default is None.
            wt_protein (str): wt sequence as a string. Must provide one of wt_protein or wt_fasta.
            wt_fasta (str): path to fasta file containing wt sequence.
                Must provide one of wt_protein or wt_fasta.
            verbose (bool): whether to print verbose output. Default is False.
            random_seed (int): random seed for reproducibility. Default is None.
        Raises:
            ValueError: if `n_steps` < 1.
            ValueError: if neither `wt_protein` nor `wt_fasta` is provided.
            ValueError: if a fasta file is passed to `wt_protein` argument.
            ValueError: if `output` is not one of 'best', 'last' or 'all'.
            ValueError: if no experts are provided.
            ValueError: if any of the preserved regions are < 1 amino acid long.
        """
        self.experts = experts
        self.parallel_chains = parallel_chains
        self.n_steps = n_steps
        self.max_mutations = max_mutations
        self.output = output
        self.preserved_regions = preserved_regions
        self.wt_protein = wt_protein
        self.wt_fasta = wt_fasta
        self.verbose = verbose
        self.random_seed = random_seed
        self.device = self.experts[0].device
        
        # Checks
        if self.n_steps < 1:
            raise ValueError("`n_steps` must be >= 1")
        if not (self.wt_protein is not None or self.wt_fasta is not None):
            raise ValueError("Must provide one of `wt_protein` or `wt_fasta`")
        if output not in ['best', 'last', 'all']:
            raise ValueError("`output` must be one of 'best', 'last' or 'all'")
        if len(self.experts) < 1:
            raise ValueError("Must provide at least one expert")
        
        if random_seed is not None:
            utils.set_seed(random_seed)
        if self.preserved_regions is not None:
            for start, end in self.preserved_regions:
                if end - start < 1:
                    raise ValueError("Preserved regions must be at least 1 amino acid long")
                
        # maintains a tokenizer with canonical alphabet
        # for the one-hot encoded chains
        self.canonical_chain_tokenizer = tokenizers.OneHotTokenizer(
                            alphabet=utils.CANONICAL_ALPHABET)
        
        if self.wt_protein is not None:
            if '.fasta' in self.wt_protein:
                raise ValueError("Did you mean to use the `wt_fasta` argument instead of `wt_protein`?")    
            self.wtseq = self.wt_protein
            # Add spaces between each amino acid if necessary
            if ' ' not in self.wtseq:
                self.wtseq = ' '.join(self.wtseq)
        # Check if wt_protein is a fasta file
        elif self.wt_fasta is not None:
            with open(Path(self.wt_fasta), 'r') as f:
                for line in f:
                    if line[0] != '>':
                        self.wtseq = line.strip()
                        # Add a space between each amino acid
                        self.wtseq = ' '.join(self.wtseq)
                        break          
        if self.verbose:
            print(f">Wildtype sequence: {self.wtseq}")
        self.reset()

        ### Hyperparams
        self.max_pas_path_length = 2


    def reset(self):
        """Initialize the parallel chains of protein sequences.
        """
        if self.random_seed is not None:
            utils.set_seed(self.random_seed)
        
        # the current state of each chain in string form
        self.chains = [self.wtseq] * self.parallel_chains
        # the current state of each chain in one-hot form
        self.chains_oh = self.canonical_chain_tokenizer(self.chains).to(self.device)
        # the WT protein in one-hot form
        self.wt_oh = self.chains_oh[0]
        # a List for storing the history of the product of experts values per chain
        self.PoE_history = []
        # a List for storing the history of one-hot encoded chains
        self.chains_oh_history = []

        for expert in self.experts:
            expert.set_wt_score(self.wtseq)

        

    def _product_of_experts(self, inputs: List[str]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Compute the product of experts.
        Computes each expert score, multiplies it by
        the expert temperature, and aggregates the scores
        by summation.
               
        Args:
            inputs: list of protein sequences of len [parallel_chains]
        Returns:
            ohs (List[torch.Tensor]): list of one-hot encoded sequences of len [parallel_chains]
            PoE (torch.Tensor): product of experts score of shape [parallel_chains]
        """
        ohs = []
        scores = []
        for expert in self.experts:
            oh, score = expert(inputs)
            ohs += [oh]
            scores += [expert.temperature * score]
        # sum scores over experts
        return ohs, torch.stack(scores, dim=0).sum(dim=0)
    

    def _compute_gradients(self, ohs: List[torch.Tensor], PoE: torch.Tensor) -> torch.Tensor:
        """Compute the gradients of the product of experts
        with respect to the one-hots. We put each experts one-hot input 
        sequence in a canonical order before summing gradients together. 

        Args:
            ohs (List[torch.Tensor]): tensor one-hot embeddings of shape [parallel_chains, seq_len, vocab_size].
                 List is of length # experts
            PoE (torch.Tensor): product of experts score of shape [parallel_chains]
        """
        # sum over chains
        oh_grads = torch.autograd.grad(PoE.sum(), ohs)
        # For each oh_grad, canonicalize the grad
        summed_grads = []
        for expert, oh_grad in zip(self.experts, oh_grads):
            # some tokenizers add a <start> and <end> token to the protein
            # sequence, check and remove here if necessary.
            # This checks whether the gradient sequence length
            # is exactly two more than the input sequence length.
            if oh_grad.shape[1] == self.chains_oh.shape[1] + 2:
                oh_grad = oh_grad[:,1:-1]
            summed_grads += [ oh_grad @ expert.expert_to_canonical_order ]
        # sum over experts
        return torch.stack(summed_grads, dim=0).sum(dim=0)
        

    def __call__(self) -> Tuple[List[str], np.ndarray]:
        """
        Run the gradient-based MCMC sampler.

        Returns:
            variants (List[str]): list of protein sequences
            scores (np.ndarray): the product of expert scores for the variants
        """
        x_rank = len(self.chains_oh.shape)-1
        seq_len = self.chains_oh.shape[-2]
        cur_chains_oh = self.chains_oh.clone()
        pos_mask = torch.zeros_like(cur_chains_oh).to(cur_chains_oh.device)
        if self.preserved_regions is not None:
            for min_pos, max_pos in self.preserved_regions:
                pos_mask[:,min_pos:max_pos+1] = 1
        pos_mask = pos_mask.bool()
        pos_mask = pos_mask.reshape(self.parallel_chains,-1)

        for i in range(self.n_steps):
            ###### sample path length
            U = torch.randint(1, 2 * self.max_pas_path_length, size=(self.parallel_chains,1))
            max_u = int(torch.max(U).item())
            u_mask = torch.arange(max_u).expand(self.parallel_chains, max_u) < U
            u_mask = u_mask.float().to(cur_chains_oh.device)

            onehot_Idx = []
            traj_list = []
            forward_categoricals = []

            # Need to use the string version of the chain to pass to experts
            ohs, PoE = self._product_of_experts(self.chains)
            grad_x = self._compute_gradients(ohs, PoE)

            # do U intermediate steps
            with torch.no_grad():
                for step in range(max_u):
                    
                    # this sampler permits identity substitions, and is
                    # likely to occur when the score change is negative for 
                    # almost all other mutations.
                    score_change = grad_x - (grad_x * cur_chains_oh).sum(-1).unsqueeze(-1)
                    traj_list += [cur_chains_oh]
                    approx_forward_expert_change = score_change.reshape(self.parallel_chains,-1) / 2
                    
                    if self.max_mutations > 0:
                        # compute the mut_distance between cur_chains_oh and wt
                        dist = utils.mut_distance(cur_chains_oh, self.wt_oh)
                        # if dist == threshold, only valid next mutations 
                        # are substitutions that reduce the mut_distance.
                        mask_flag = (dist == self.max_mutations).bool()
                        mask_flag = mask_flag.reshape(self.parallel_chains)
                        mask = utils.mutation_mask(cur_chains_oh, self.wt_oh)
                        mask = mask.reshape(self.parallel_chains,-1)
                        # Apply mask to constrain proposals within edit distance of WT
                        mask[~mask_flag] = False
                        approx_forward_expert_change[mask] = -np.inf
                    
                    # Apply mask to avoid changing preserved regions
                    approx_forward_expert_change[pos_mask] = -np.inf

                    cd_forward = torch.distributions.one_hot_categorical.OneHotCategorical(
                        probs=utils.safe_logits_to_probs(approx_forward_expert_change))
                    forward_categoricals += [cd_forward]
                    changes_all = cd_forward.sample((1,)).squeeze(0)
                    onehot_Idx += [changes_all]
                    changes_all = changes_all.view(self.parallel_chains, seq_len, -1)
                    row_select = changes_all.sum(-1).unsqueeze(-1)  # [n_chains,seq_len,1]
                    new_x = cur_chains_oh * (1.0 - row_select) + changes_all
                    cur_u_mask = u_mask[:, step].unsqueeze(-1).unsqueeze(-1)
                    cur_chains_oh = cur_u_mask * new_x + (1 - cur_u_mask) * cur_chains_oh
                    
                y = cur_chains_oh
            
            # last step
            y_strs = self.canonical_chain_tokenizer.decode(y)
            ohs, proposed_PoE = self._product_of_experts(y_strs)
            grad_y = self._compute_gradients(ohs, proposed_PoE)
            grad_y = grad_y.detach()

            with torch.no_grad():
                 # backwd from y -> x
                traj_list.append(y)
                traj = torch.stack(traj_list[1:], dim=1)
                reverse_score_change = grad_y.unsqueeze(1) - (grad_y.unsqueeze(1) * traj).sum(-1).unsqueeze(-1)
                reverse_score_change = reverse_score_change.reshape(self.parallel_chains, max_u, -1) / 2.0
                log_ratio = 0
                for id in range(len(onehot_Idx)):
                    cd_reverse = torch.distributions.one_hot_categorical.OneHotCategorical(
                        probs=utils.safe_logits_to_probs(reverse_score_change[:,id]))
                    log_ratio += u_mask[:,id] * (cd_reverse.log_prob(onehot_Idx[id]) - forward_categoricals[id].log_prob(onehot_Idx[id]))
                
                #log_acc = log_backwd - log_fwd
                m_term = (proposed_PoE - PoE)
                log_acc = m_term + log_ratio
                
                accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
                cur_chains_oh = y * accepted + (1.0 - accepted) * cur_chains_oh

            # Current chain state book-keeping    
            self.chains_oh = cur_chains_oh
            self.chains = self.canonical_chain_tokenizer.decode(cur_chains_oh)
            # History book-keeping
            self.chains_oh_history += [cur_chains_oh.clone()]
            PoE = proposed_PoE * accepted.squeeze() + PoE * (1. - accepted.squeeze())
            self.PoE_history += [PoE.clone()]
            
            if self.verbose:
                x_strs = self.canonical_chain_tokenizer.decode(cur_chains_oh)
                print(f'step {i} acceptance rate: {log_acc.exp().item():.4f}')
                for idx,variant in enumerate(x_strs):
                    print(f'>chain {idx}, Product of Experts score: {PoE[idx]:.4f}')
                    utils.print_variant_in_color(variant, self.wtseq)

            if self.max_mutations > 0:
                # Once a chain reaches the max mutations, reset it to WT            
                dist = utils.mut_distance(cur_chains_oh, self.wt_oh)
                mask_flag = (dist >= self.max_mutations).bool()
                mask_flag = mask_flag.reshape(self.parallel_chains)
                cur_chains_oh[mask_flag] = self.wt_oh            

        if self.output == 'last':
            output_ = self.canonical_chain_tokenizer.decode(cur_chains_oh)
            scores_ = self.PoE_history[-1].detach().cpu().numpy()
        elif self.output == 'all':
            output_ = []
            for i in range(len(self.chains_oh_history)):
                output_ += [ self.canonical_chain_tokenizer.decode(self.chains_oh_history[i]) ]
            scores_ = torch.stack(self.PoE_history).detach().cpu().numpy()
        elif self.output == 'best':
            best_idxs = torch.stack(self.PoE_history).argmax(0)
            chains_oh_history = torch.stack(self.chains_oh_history) # [n_steps, n_chains, seq_len, n_tokens]
            output_ = self.canonical_chain_tokenizer.decode(
                torch.stack([chains_oh_history[best_idxs[i],i] for i in range(self.parallel_chains)]))
            scores_ = torch.stack(
                [self.PoE_history[best_idxs[i]][i] for i in range(self.parallel_chains)]).detach().cpu().numpy()
        return output_, scores_


