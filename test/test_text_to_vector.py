from so_classifier.text_to_vector import my_bag_of_words
import unittest


class TestTextVector(unittest.TestCase):

    def test_my_bag_of_words(self):
        words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
        examples = ['hi how are you']
        answers = [[1, 1, 0, 1]]
        self.assertFalse(True)
        for ex, ans in zip(examples, answers):
            self.assertFalse((my_bag_of_words(ex, words_to_index, 4) != ans).any())


if __name__ == '__main__':
    unittest.main()

