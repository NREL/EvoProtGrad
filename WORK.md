# TODOs

- Upgrade transformers >= 4.36.0

## EsmTherm

This unsupervised model https://github.com/SimonKitSangChu/EsmTherm/blob/f7d8c7a7c705859c51a4cb06e71043733dafe560/evaluate.py#L36 appears to be just a fine-tuned version of the EsmForMaskedLM `facebook/esm2_t6_8M_UR50D` model.
An easy way to use this as an expert would be to upload the  fine-tuned model checkpoints that the authors provide to HuggingFace, after which they can be automatically used in EvoProtGrad by specifying, for example:

esmtherm_expert = evo_prot_grad.get_expert(‘esmtherm’, model = EsmForMaskedLM.from_pretrained("NREL/esmtherm_t6_8M_UR50D"), …)


We would need to add a new expert for “EsmForSequenceClassification” to support this version https://github.com/SimonKitSangChu/EsmTherm/blob/f7d8c7a7c705859c51a4cb06e71043733dafe560/evaluate.py#L63C15-L63C15 , which would be easy enough but would involve a Pull Request to add the new esm expert type.
 
 ## mutation scoring

 As for integrating this into EvoProtGrad, it may need some refactoring as we would need to decouple the HuggingFaceExpert and the mutation scoring algorithm it uses. Currently, these are tightly coupled.
 
I like the idea of creating a file named something like `evo_prot_grad/common/mutation_scoring.py` where we could implement different scoring functions. Then, each HuggingFaceExpert could be paired with the mutation scoring method of choice via a `scoring_strategy` argument (similar to https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py). I’d have to think a bit about what the function signature for each mutation scoring method would be..

- https://github.com/Amelie-Schreiber/protein_mutation_scoring/blob/main/scoring_esm2.py 
