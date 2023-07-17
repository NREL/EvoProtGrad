import unittest 
from evo_prot_grad import get_expert
from evo_prot_grad.models import EVCouplings
from transformers import AutoModel


class TestExperts(unittest.TestCase):

    def setUp(self):
        self.sequence = ['S K G E E L F T G V V P I L V E L D G D V N G H K F S V S G E G E G D A T Y G K L T L K F I C T T G K L P V P W P T L V T T L S Y G V Q C F S R Y P D H M K Q H D F F K S A M P E G Y V Q E R T I F F K D D G N Y K T R A E V K F E G D T L V N R I E L K G I D F K E D G N I L G H K L E Y N Y N S H N V Y I M A D K Q K N G I K V N F K I R H N I E D G S V Q L A D H Y Q Q N T P I G D G P V L L P D N H Y L S T Q S A L S K D P N E K R D H M V L L E F V T A A G I T H G M D E L Y K']

    
    def test_get_expert_pretrained(self):
        """
        Test that get_expert works with all experts.
        """
        experts = ['esm', 'bert', 'causallm']

        for expert in experts:
            e = get_expert(expert, use_without_wildtype=True)

            out = e(self.sequence)

    def test_get_expert_custom(self):
        evcouplings_model = EVCouplings(
           'test/PABP_YEAST.model_params',
           'test/gfp.fasta')
        e = get_expert('evcouplings', 1.0,
                       evcouplings_model,
                       use_without_wildtype=True)
        out = e(self.sequence)

        onehot_model = AutoModel.from_pretrained('NREL/avGFP-fluorescence-onehot-cnn', trust_remote_code=True)

        e = get_expert(
                'onehot_downstream_regression', 1.0,
                onehot_model,
                use_without_wildtype=True)
        out = e(self.sequence)
