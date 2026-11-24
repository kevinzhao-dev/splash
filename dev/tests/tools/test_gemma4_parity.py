import unittest

from dev.tools import gemma4_parity


class Gemma4ParityTest(unittest.TestCase):
    def test_identical_logits_and_tokens_agree_exactly(self):
        logits = [float(i % 17) - 8.0 for i in range(64)]
        scores = gemma4_parity.compare(logits, logits, [3, 4], [3, 4])
        self.assertEqual(scores["max_abs_logit_error"], 0.0)
        self.assertEqual(scores["rms_logit_error"], 0.0)
        self.assertEqual(scores["top10_overlap"], 10)
        self.assertEqual(scores["greedy_common_prefix"], 2)
        self.assertTrue(scores["greedy_identical"])

    def test_a_diverging_answer_counts_its_common_prefix(self):
        splash = [0.0] * 32
        reference = [0.0] * 32
        splash[5], reference[5] = 2.0, 1.0
        scores = gemma4_parity.compare(splash, reference, [1, 2, 3], [1, 9, 3])
        self.assertEqual(scores["max_abs_logit_error"], 1.0)
        self.assertAlmostEqual(scores["rms_logit_error"], (1.0 / 32) ** 0.5)
        self.assertEqual(scores["greedy_common_prefix"], 1)
        self.assertFalse(scores["greedy_identical"])

    def test_the_chat_prompt_is_one_user_turn_before_an_empty_thought(self):
        prompt = gemma4_parity.chat_prompt("  Hi.\n")
        self.assertEqual(
            prompt,
            "<|turn>user\nHi.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        )


if __name__ == "__main__":
    unittest.main()
