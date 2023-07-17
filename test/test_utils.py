import unittest 
import torch
from transformers import AutoModel
from evo_prot_grad import get_expert
from evo_prot_grad import DirectedEvolution
import evo_prot_grad.common.utils as utils
import numpy as np

class TestUtils(unittest.TestCase):
    
    def test_safe_logits_to_probs(self):
        logits = torch.randn(2, 10, 21)
        probs = utils.safe_logits_to_probs(logits)
        # test shape
        self.assertEqual(probs.shape, logits.shape)
        # test sum to 1
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(2, 10)))

    def test_mut_distance(self):
        """ Test edit distance function for one-hot encoded sequences."""
        x = torch.zeros(2, 10, 21)
        wt = torch.zeros(1, 10, 21)
        # test no difference
        self.assertTrue(torch.allclose(utils.mut_distance(x, wt), torch.zeros(2)))
        # test one difference
        x[0, 0, 1] = 1
        self.assertTrue(torch.allclose(utils.mut_distance(x, wt), torch.tensor([1., 0.])))
        # test multiple differences
        x[0, 1, 2] = 1
        x[0, 2, 3] = 1
        self.assertTrue(torch.allclose(utils.mut_distance(x, wt), torch.tensor([3., 0.])))

    def test_mutation_mask(self):
        """ Test mutation mask function for one-hot encoded sequences."""
        x = torch.zeros(2, 10, 21)
        wt = torch.zeros(1, 10, 21)
        # test no difference
        self.assertTrue(torch.equal(utils.mutation_mask(x, wt), torch.ones(2, 10, 21).bool()))
        # test one difference
        wt[0, 0, 1] = 1
        mask = torch.ones(2, 10, 21).bool()
        mask[0, 0, 1] = False
        mask[1, 0, 1] = False
        self.assertTrue(torch.equal(utils.mutation_mask(x, wt), mask))

    def test_expert_alphabet_to_canonical(self):
        """ Test expert alphabet to canonical alphabet function."""
        expert_alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        alignment_matrix = utils.expert_alphabet_to_canonical(expert_alphabet, 'cpu')
        # test shape
        self.assertEqual(alignment_matrix.shape, (len(expert_alphabet), len(utils.CANONICAL_ALPHABET)))
        # test identity
        self.assertTrue(torch.allclose(alignment_matrix, torch.eye(len(expert_alphabet))))
        # test reverse
        expert_alphabet = expert_alphabet[::-1]
        alignment_matrix = utils.expert_alphabet_to_canonical(expert_alphabet, 'cpu')
        # rotate identity matrix 90
        x = torch.rot90(torch.eye(len(expert_alphabet)), 1, [0, 1])
        self.assertTrue(torch.allclose(alignment_matrix,  x))


    def test_seed(self):
        """ Test seed function."""
        gfp_expert = get_expert(
            'onehot_downstream_regression',
            1.0,
            AutoModel.from_pretrained('NREL/avGFP-fluorescence-onehot-cnn',
                                       trust_remote_code=True))
        epg = DirectedEvolution(
            experts=[gfp_expert],
            parallel_chains=5,
            n_steps=10,
            max_mutations=-1,
            output = 'last',
            wt_fasta='test/gfp.fasta',
            random_seed=42,
            verbose=False
        )
        self.assertEqual(torch.initial_seed(), 42)

        out1, score1 = epg()
        epg.reset()
        out2, score2 = epg()
        self.assertTrue( out1 == out2 )
        self.assertTrue( np.allclose(score1, score2) )

        # test different seed
        epg = DirectedEvolution(
            experts=[gfp_expert],
            parallel_chains=5,
            n_steps=10,
            max_mutations=-1,
            output = 'last',
            wt_fasta='test/gfp.fasta',
            random_seed=43,
            verbose=False
        )
        self.assertEqual(torch.initial_seed(), 43)

        out3, score3 = epg()
        self.assertTrue( out1 != out3 )
        self.assertFalse( np.allclose(score1, score3) )