First, import our library:

```python
import evo_prot_grad
```

Create a `ProtBERT` expert from a pretrained 🤗 HuggingFace protein language model (PLM) using `evo_prot_grad.get_expert`:

```python
prot_bert_expert = evo_prot_grad.get_expert('bert', temperature = 1.0, device = 'cuda')
```
The default BERT-style PLM in `EvoProtGrad` is `Rostlab/prot_bert`. Normally, we would need to also specify the model and tokenizer. When using a default PLM expert, we automatically pull these from the HuggingFace Hub. The temperature parameter rescales the expert scores and can be used to trade off the importance of different experts. For masked language models like `prot_bert`, we score variant sequences with the sum of amino acid log probabilities by default.

Then, we create an instance of `DirectedEvolution` and run the search, returning a list of the best variant per Markov chain (as measured by the `prot_bert` expert):

```python
variants, scores = evo_prot_grad.DirectedEvolution(
                   wt_fasta = 'test/gfp.fasta',    # path to wild type fasta file
                   output = 'best',                # return best, last, all variants    
                   experts = [prot_bert_expert],   # list of experts to compose
                   parallel_chains = 1,            # number of parallel chains to run
                   n_steps = 20,                   # number of MCMC steps per chain
                   max_mutations = 10,             # maximum number of mutations per variant
                   verbose = True                  # print debug info to command line
)()
```

This class implements PPDE, the gradient-based discrete MCMC sampler introduced in our [paper](https://doi.org/10.1088/2632-2153/accacd).

### Specifying the model and tokenizer

To load a HuggingFace expert with a specific model and tokenizer, provide them as arguments to `get_expert`:

```python
from transformers import AutoTokenizer, EsmForMaskedLM

esm2_expert = evo_prot_grad.get_expert(
                'esm',
                model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D"),
                tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D"),
                temperature = 1.0,
                device = 'cuda')
```

### Composing 2+ Experts

You can compose multiple experts by passing multiple experts to `DirectedEvolution` as a list. As an example, we provide a ConvNet-based expert that predicts the fluorescence of GFP variants in the HuggingFace Hub:

```python 

import evo_prot_grad
from transformers import AutoModel

prot_bert_expert = evo_prot_grad.get_expert('bert', temperature = 1.0, device = 'cuda')

# onehot_downstream_regression are experts that predict a downstream scalar property
# from a one-hot encoding of the protein sequence
fluorescence_expert = evo_prot_grad.get_expert(
                        'onehot_downstream_regression',
                        temperature = 1.0,
                        model = AutoModel.from_pretrained('NREL/avGFP-fluorescence-onehot-cnn',
                                                          trust_remote_code=True),
                        device = 'cuda')

variants, scores = evo_prot_grad.DirectedEvolution(
                        wt_fasta = 'test/gfp.fasta',
                        output = 'best',
                        experts = [prot_bert_expert, fluorescence_expert],
                        parallel_chains = 1,
                        n_steps = 100,              
                        max_mutations = 10
)()
```