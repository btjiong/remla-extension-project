from so_classifier.text_preprocessing import text_prepare
import unittest


class TestTextPrepare(unittest.TestCase):

    def test_simple_text_prepare(self):
        examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                    "How to free c++ memory vector<int> * arr?"]
        answers = ["sql server equivalent excels choose function",
                   "free c++ memory vectorint arr"]
        for ex, ans in zip(examples, answers):
            self.assertEqual(text_prepare(ex), ans)


if __name__ == '__main__':
    unittest.main()




