import unittest 
from transformers import AutoModel
from evo_prot_grad import get_expert
from evo_prot_grad.common.sampler import DirectedEvolution
import numpy as np

class TestSampler(unittest.TestCase):
    
    def setUp(self):
        self.gfp_expert = get_expert(
            'onehot_downstream_regression',
            1.0,
            AutoModel.from_pretrained('NREL/avGFP-fluorescence-onehot-cnn',
                                       trust_remote_code=True))

    def test_init(self):
        # Test that the sampler is initialized correctly
        epg = DirectedEvolution(
            experts=[self.gfp_expert],
            parallel_chains=2,
            n_steps=1,
            max_mutations=-1,
            output = 'best',
            wt_fasta='test/gfp.fasta'
        )


    def test_preserved_regions(self):
        # Test no preserved regions
        # Test one preserved region
        # Test multiple preserved regions
        # Test with parallel chains
        # Test with overlapping regions
        preserved_regions = [
            [],
            [(2,3)],
            [(2,3), (20,30)],
            [(20,30), (25,35)],
        ]
        for preserved_region in preserved_regions:
            epg = DirectedEvolution(
                experts=[self.gfp_expert],
                parallel_chains=2,
                n_steps=1,
                max_mutations=-1,
                output = 'best',
                wt_fasta='test/gfp.fasta',
                preserved_regions=preserved_region
            )()
        # Test error if start > end
        with self.assertRaises(ValueError):
            epg = DirectedEvolution(
                experts=[self.gfp_expert],
                parallel_chains=2,
                n_steps=1,
                max_mutations=-1,
                output = 'best',
                wt_fasta='test/gfp.fasta',
                preserved_regions=[(3,2)]
            )()


    def test_maximum_mutations(self):
        # Test no maximum mutations
        # Test maximum mutations
        # Test maximum mutations with preserved regions
        pr = [None, None, [(20,30)]]
        max_mutations = [-1, 1000, 1]
        for pr_, max_mutation in zip(pr, max_mutations):
            epg = DirectedEvolution(
                experts=[self.gfp_expert],
                parallel_chains=2,
                n_steps=1,
                max_mutations=max_mutation,
                output = 'best',
                wt_fasta='test/gfp.fasta',
                preserved_regions=pr_
            )()


    def test_n_steps(self):
        # Test n_steps = 1
        # Test n_steps <= 0
        epg = DirectedEvolution(
            experts=[self.gfp_expert],
            parallel_chains=2,
            n_steps=1,
            max_mutations=-1,
            output = 'best',
            wt_fasta='test/gfp.fasta'
        )()

        # Assert n_steps <= 0 raises error
        with self.assertRaises(ValueError):
            epg = DirectedEvolution(
                experts=[self.gfp_expert],
                parallel_chains=2,
                n_steps=0,
                max_mutations=-1,
                output = 'best',
                wt_fasta='test/gfp.fasta'
            )()


    def test_wildtype(self):
        # Test loading wildtype from file
        # Test loading wildtype from string
        wt = 'S K G E E L F T G V V P I L V E L D G D V N G H K F S V S G E G E G D A T Y G K L T L K F I C T T G K L P V P W P T L V T T L S Y G V Q C F S R Y P D H M K Q H D F F K S A M P E G Y V Q E R T I F F K D D G N Y K T R A E V K F E G D T L V N R I E L K G I D F K E D G N I L G H K L E Y N Y N S H N V Y I M A D K Q K N G I K V N F K I R H N I E D G S V Q L A D H Y Q Q N T P I G D G P V L L P D N H Y L S T Q S A L S K D P N E K R D H M V L L E F V T A A G I T H G M D E L Y K'
        epg = DirectedEvolution(
            experts=[self.gfp_expert],
            parallel_chains=2,
            n_steps=1,
            max_mutations=-1,
            output = 'best',
            wt_protein=wt
        )
        self.assertEqual(epg.wtseq, wt)


    def test_output(self):
        """Test that the output is correct
           for 'best', 'last', 'all'
        """
        # Test 'best'
        epg = DirectedEvolution(
            experts=[self.gfp_expert],
            parallel_chains=2,
            n_steps=1,
            max_mutations=-1,
            output = 'best',
            wt_fasta='test/gfp.fasta'
        )()
        self.assertEqual(len(epg), 2)
        variants, scores = epg
        self.assertEqual(len(variants), 2)
        self.assertEqual(len(scores), 2)
        self.assertTrue(isinstance(scores[0], np.float32), "Score is not a float, {}".format(type(scores[0])))

        # Test 'last'
        epg = DirectedEvolution(
            experts=[self.gfp_expert],
            parallel_chains=2,
            n_steps=1,
            max_mutations=-1,
            output = 'last',
            wt_fasta='test/gfp.fasta'
        )()
        self.assertEqual(len(epg), 2)

        # Test 'all'
        epg = DirectedEvolution(
            experts=[self.gfp_expert],
            parallel_chains=2,
            n_steps=2,
            max_mutations=-1,
            output = 'all',
            wt_fasta='test/gfp.fasta'
        )()
        self.assertEqual(len(epg), 2)
        self.assertEqual(len(epg[0]), 2)