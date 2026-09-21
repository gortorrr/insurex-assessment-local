import unittest

from insurex.rag.ingestion import _token_count


class IngestionTokenCountTests(unittest.TestCase):
    def test_nested_tokenizer_output_counts_tokens_not_batch_keys(self):
        class FakeTokenizer:
            def __call__(self, text):
                return {"input_ids": [[1, 2, 3, 4]]}

        self.assertEqual(_token_count("x", FakeTokenizer()), 4)

    def test_tensor_like_shape_uses_last_dimension(self):
        class Shape:
            shape = (1, 7)

        class FakeTokenizer:
            def __call__(self, text):
                return {"input_ids": Shape()}

        self.assertEqual(_token_count("x", FakeTokenizer()), 7)
