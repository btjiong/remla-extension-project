"""
    Tests the text_to_vector class
"""
import unittest

from so_classifier.model.text_to_vector import my_bag_of_words


class TestTextVector(unittest.TestCase):
    """
    Tests the text_to_vector class
    """

    def test_my_bag_of_words(self):
        """
        A basic test , checking if a piece of text gets converted to a vector correctly.
        """
        words_to_index = {"hi": 0, "you": 1, "me": 2, "are": 3}
        examples = ["hi how are you"]
        answers = [[1, 1, 0, 1]]
        for ex, ans in zip(examples, answers):
            self.assertFalse((my_bag_of_words(ex, words_to_index, 4) != ans).any())


if __name__ == "__main__":
    unittest.main()
