import unittest 
import torch 
from evo_prot_grad import get_expert
from transformers import AutoModel


class TestVariantScoring(unittest.TestCase):

    def setUp(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence = 'S K G E E L F T G V V P I L V E L D G D V N G H K F S V S G E G E G D A T Y G K L T L K F I C T T G K L P V P W P T L V T T L S Y G V Q C F S R Y P D H M K Q H D F F K S A M P E G Y V Q E R T I F F K D D G N Y K T R A E V K F E G D T L V N R I E L K G I D F K E D G N I L G H K L E Y N Y N S H N V Y I M A D K Q K N G I K V N F K I R H N I E D G S V Q L A D H Y Q Q N T P I G D G P V L L P D N H Y L S T Q S A L S K D P N E K R D H M V L L E F V T A A G I T H G M D E L Y K'

    def test_mutant_marginal_parallel_chains_1(self):
        expert = get_expert('esm', 'mutant_marginal', 1.0, device=self.device)
        expert.init_wildtype(self.sequence)

        seqs = [self.sequence]
        self.assertEqual( len(seqs) , 1 )  # parallel_chains is 1

        x_oh, mutant_marginal_score = expert(seqs)
        self.assertEqual( torch.zeros(1).to(mutant_marginal_score.device) , mutant_marginal_score )

    def test_mutant_marginal_parallel_chains_3(self):
        expert = get_expert('esm', 'mutant_marginal', 1.0, device=self.device)
        expert.init_wildtype(self.sequence)

        seqs = [self.sequence]*3
        self.assertEqual( len(seqs) , 3 )

        x_oh, mutant_marginal_score = expert(seqs)
        # floating point rounding error introduced by subtracting two ~equal numbers,
        # so need to use allclose here.
        self.assertTrue( torch.allclose(torch.zeros(3).to(mutant_marginal_score.device),
                                      mutant_marginal_score,  atol = 1e-4 ) )

    def test_pseudolikelihood_ratio_parallel_chains_1(self):
        expert = get_expert('esm', 'pseudolikelihood_ratio', 1.0, device=self.device)
        expert.init_wildtype(self.sequence)

        seqs = [self.sequence]
        self.assertEqual( len(seqs) , 1 )

        x_oh, pll_ratio = expert(seqs)
        self.assertEqual( torch.zeros(1).to(pll_ratio.device) , pll_ratio )
    
    def test_pseudolikelihood_ratio_parallel_chains_3(self):
        expert = get_expert('esm', 'pseudolikelihood_ratio', 1.0, device=self.device)
        expert.init_wildtype(self.sequence)

        seqs = [self.sequence]*3
        self.assertEqual( len(seqs) , 3 )

        x_oh, pll_ratio = expert(seqs)
        self.assertTrue( torch.allclose(torch.zeros(3).to(pll_ratio.device),
                                      pll_ratio,  atol = 1e-4 ), pll_ratio )
        
    def test_attribute_value_parallel_chains_1(self):
        expert = get_expert(
                'onehot_downstream_regression',
                'attribute_value',
                1.0,
                AutoModel.from_pretrained('NREL/avGFP-fluorescence-onehot-cnn',
                                        trust_remote_code=True),
                device=self.device)
        expert.init_wildtype(self.sequence)

        seqs = [self.sequence]
        self.assertEqual( len(seqs) , 1 )

        x_oh, attribute_value = expert(seqs)
        self.assertEqual( torch.zeros(1).to(attribute_value.device) , attribute_value )
    
    def test_attribute_value_parallel_chains_3(self):
        expert = get_expert(
                'onehot_downstream_regression',
                'attribute_value',
                1.0,
                AutoModel.from_pretrained('NREL/avGFP-fluorescence-onehot-cnn',
                                        trust_remote_code=True),
                device=self.device)
        expert.init_wildtype(self.sequence)

        seqs = [self.sequence]*3
        self.assertEqual( len(seqs) , 3 )

        x_oh, attribute_value = expert(seqs)
        self.assertTrue( torch.allclose(torch.zeros(3).to(attribute_value.device),
                                      attribute_value , atol = 1e-4 ), attribute_value )
        
