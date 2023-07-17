In EvoProtGrad, an expert has a tokenizer for converting lists of strings into Torch one-hot tensors and a differentiable model that takes as input a one-hot-encoded protein sequence and returns a score. The idea is to combine multiple experts together for directed evolution, since some experts may correlate with the desired attribute(s) you wish to optimize while other experts help steer search away from deleterious mutations. We provide a few  experts in `evo_prot_grad/experts` that you can use out of the box, such as:

Protein Language Models (PLMs)

- `bert`, BERT-style PLMs, default: `Rostlab/prot_bert`
- `causallm`, CausalLM-style PLMs, default: `lightonai/RITA_s`
- `esm`, ESM-style PLMs, default: `facebook/esm2_t6_8M_UR50D`

Potts models

- `evcouplings`

and an generic expert for supervised downstream regression models

- `onehot_downstream_regression`


## What is a Product of Experts?

`EvoProtGrad` combines multiple experts into a Product of Experts.
Formally, each expert defines a (typically unnormalized) probability distribution over protein sequences. High probability can correspond to high fitness or high [evolutionary density](https://www.nature.com/articles/s41587-021-01146-5). 
A Product of Experts is the product of these distributions, normalized by an unknown partition function $Z$.
For example, if we have two experts $F$ and $G$, the Product of Experts is:

$$
P(X) = \frac{1}{Z}  F(X) G(X)^{\lambda}
$$


where $F(X)$ may correspond to the probability given by an unsupervised pretrained PLM to sequence $X$ and $G(X)$ may correspond to the probability assigned to $X$ by a supervised downstream regression model (interpreted as an unnormalized Boltzmann distribution with temperature $\lambda$). We can compute the sum of the (log) probabilities of the experts as follows:

$$
\log P(X) = \log F(X) - \lambda \log G(X) - \log Z.
$$

In `EvoProtGrad`, the "score" of each expert corresponds to either $\log F(X)$ or $\log G(X)$ here. In most cases, we interpret the scalar output of a neural network as the score.
The magic of the Product of Experts formulation is that it enables us to compose arbitrary numbers of experts, essentially allowing us to "plug and play" with different experts to guide the search.

In actuality, instead of just searching for a protein variant that maximizes $P(X)$, `EvoProtGrad` uses gradient-based discrete MCMC to *sample* from $P(X)$.
MCMC is necessary for sampling from $P(X)$ because it is impractical to compute the partition function $Z$ exactly.
Uniquely to `EvoProtGrad`, as long *all* experts are *differentiable*, our sampler can use the gradient of $\log F(X) - \lambda \log G(X)$ with respect to the one-hot protein $X$ to identify the most promising mutation to apply to $X$, which vastly speeds up MCMC convergence.

##  🤗 HuggingFace Transformers 

`EvoProtGrad` provides a convenient interface for defining and using experts from the HuggingFace Hub.
To use pretrained PLMs from the HuggingFace Hub with gradient-based discrete MCMC, we swap out each transformer's token embedding layer for a custom [one-hot token embedding layer](https://nrel.github.io/EvoProtGrad/api/common/embeddings/#onehotembedding). This enables us to compute and access gradients with respect to one-hot input protein sequences.

We provide a baseclass `evo_prot_grad.experts.base_experts.HuggingFaceExpert` which is subclassed to support various types of HuggingFace PLMs. Currently, we provide three subclasses for

- BERT-style PLMs (`evo_prot_grad.experts.bert_expert.BertExpert`)
- CausalLM-style PLMs (`evo_prot_grad.experts.causallm_expert.CausalLMExpert`)
- ESM-style PLMs (`evo_prot_grad.experts.esm_expert.EsmExpert`)

Each HuggingFace PLM expert has to specify the model and tokenizer to use. We provide defaults for each type of PLM. 
For example, an ESM2 expert can be instantiated with only:

```python
esm2_expert = evo_prot_grad.get_expert('esm', temperature = 1.0, device = 'cuda')
```

with default model `EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")` and tokenizer `AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")`.
With custom model and tokenizer:

```python
from transformers import AutoTokenizer, EsmForMaskedLM

esm2_expert = evo_prot_grad.get_expert(
                'esm',
                model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D"),
                tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D"),
                temperature = 1.0,
                device = 'cuda')
```

## EVcouplings Potts

Potts models are (differentiable) linear undirected graphical models that capture pairwise interactions between amino acids in a protein sequence. 
They are fit to multiple sequence alignments (MSAs) of homologous sequences and can be used to score the evolutionary density of a sequence with respect to the MSA via the *Hamiltonian* score. 
The Hamiltonian score [has been demonstrated](https://www.nature.com/articles/s41587-021-01146-5) to be a good proxy for the fitness of a variant.
We provide an expert that wrap Potts models from the [debbiemarkslab/EVcouplings](https://github.com/debbiemarkslab/EVcouplings) library.

The EVcouplings expert expects an EVcouplings model `evo_prot_grad.models.EVCouplings` (which is a PyTorch re-implementation of the EVcouplings `Couplings` Potts model), whose parameters we initialize from a file in "plmc_v2" format produced by the `debbiemarkslab/plmc` library.

Example:

```python

evcouplings_model = EVCouplings(
            model_params_file = 'GFP_AEQVI_Sarkisyan2016-linear-15/plmc/uniref100.model_params',
            fasta_file = 'test/gfp.fasta')

evcouplings_expert = get_expert(
            'evcouplings', 
            temperature = 1.0,
            model = evcouplings_model)
```

## Downstream Regression

We provide a generic expert for supervised downstream regression models and a simple 1D-Convolutional neural network (CNN) PyTorch Module (`evo_prot_grad.models.OneHotCNN`) that predicts a scalar fitness score from a one-hot-encoded protein sequence.

To get started, we provide a pretrained OneHotCNN model trained on the [Green Fluorescent Protein (GFP) dataset](https://datadryad.org/stash/dataset/doi:10.6078/D1K71B) that can be loaded from the HuggingFace Hub:

```python
onehotcnn_model = AutoModel.from_pretrained(
            'NREL/avGFP-fluorescence-onehot-cnn', trust_remote_code=True)

regression_expert = get_expert(
            'onehot_downstream_regression',
            temperature = 1.0,
            model = onehotcnn_model)
```

### Training your own downstream regression model

You can train a `OneHotCNN` (or your own custom PyTorch model) as you normally would any supervised regression model with a dataset of pairs of (variants, fitness/stability/etc.) labels. We provide a default tokenizer class [`OneHotTokenizer`](https://nrel.github.io/EvoProtGrad/api/common/tokenizers) that can be used to convert a list of strings of amino acids into Torch one-hot tensors. Then, you can use the trained model as an expert in `EvoProtGrad`:

```python
# Load your trained model
onehotcnn_model = OneHotCNN()
onehotcnn_model.load_state_dict(torch.load('onehotcnn.pt'))

regression_expert = get_expert(
            'onehot_downstream_regression',
            temperature = 1.0,
            model = onehotcnn_model)
```

## Choosing the Expert Temperature

The expert temperature $\lambda$ controls the relative importance of the expert in the Product of Experts. By default it is set to 1. 

!!! note 

    By default, the expert's score for a variant is normalized by the wild type score, i.e., we subtract the wild type score from the variant score. This is to ensure that each expert in the Product of Experts is centered around 0. If you want to use the raw expert score, set `use_without_wildtype = True` when instantiating the expert.

If using wild type centering, then we recommend first trying temperatures of 1.0 for each $\lambda$. 
We describe a simple heuristic for selecting $\lambda$ via a grid search using a small dataset of labeled variants in Section 5.2 of our [paper](https://doi.org/10.1088/2632-2153/accacd).