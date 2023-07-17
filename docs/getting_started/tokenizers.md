Each expert requires a tokenizer. The tokenizer tells `EvoProtGrad` the particular amino acid ordering the expert model was trained to use and is called internally to convert a list of strings (the variants) into Torch tensors for the expert model. 

- We provide a default tokenizer class [`OneHotTokenizer`](https://nrel.github.io/EvoProtGrad/api/common/tokenizers) that can be used to convert a list of strings of amino acids into Torch one-hot tensors in the canonical order.
- HuggingFace experts expect a tokenizer of type `PreTrainedTokenizerBase` to be provided. 
- You can define your own custom tokenizer class by subclassing `evo_prot_grad.common.tokenizers.ExpertTokenizer` and implementing the `__call__` and `decode` methods. See `OneHotTokenizer` for an example.

## Canonicalizing amino acid sequence order

A subtlety with combining different one-hot protein sequence models is that the models may each use a different amino acid alphabet. For example, one model may use alphabet `'ACDEFGHIKLMNPQRSTVWY'` while another may have extra `<start>` and `<end>` tokens or use a different order. If the entries of the one-hot encoded protein sequences for each model do not align, we cannot sum their gradients together for gradient-based MCMC.

To address this, we define a "canonical" amino acid ordering, `'ACDEFGHIKLMNPQRSTVWY'`, and canonicalize the one-hot encoded sequences for each expert model to this ordering. Each expert internally computes and maintains a binary matrix that maps one-hot encoded tensors between the canonical ordering and the ordering used by the expert model (the binary matrix computation code is in `evo_prot_grad.common.utils.expert_alphabet_to_canonical`). This matrix is applied to Torch one-hot tensors via matrix multiplication in a differentiable manner.


