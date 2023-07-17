# evo_prot_grad.experts

---

## Expert
::: evo_prot_grad.experts.base_experts.Expert
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        members: ["_tokenize", "_get_last_one_hots", "_model_output_to_scalar_energy", "set_wt_energy", "__call__"]


## HuggingFaceExpert
::: evo_prot_grad.experts.base_experts.HuggingFaceExpert
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        members: ["_model_output_to_scalar_energy", "set_wt_energy", 
        "__call__"]

## BERTExpert
::: evo_prot_grad.experts.bert_expert.BERTExpert
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        members: ["_tokenize", "_get_last_one_hots"]

## CausalLMExpert
::: evo_prot_grad.experts.causallm_expert.CausalLMExpert
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        members: ["_tokenize", "_get_last_one_hots"]

## EsmExpert
::: evo_prot_grad.experts.esm_expert.EsmExpert
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        members: ["_tokenize", "_get_last_one_hots"]

## AttributeExpert
::: evo_prot_grad.experts.base_experts.AttributeExpert
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        members: ["_tokenize", "_model_output_to_scalar_energy", "set_wt_energy", "__call__"]

---


## EVCouplingsExpert
::: evo_prot_grad.experts.evcouplings_expert.EVCouplingsExpert
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true

---

## OneHotDownstreamExpert
::: evo_prot_grad.experts.onehot_downstream_regression_expert.OneHotDownstreamRegressionExpert
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true