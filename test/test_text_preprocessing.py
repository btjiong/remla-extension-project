"""
    Contains tests concering the preprocssing of the provided data.
"""

import unittest

from so_classifier.model.text_preprocessing import text_prepare


class TestTextPrepare(unittest.TestCase):
    """
    Contains test concering with the preprocssing of the provided data.
    """

    def test_simple_text_prepare(self):
        """
        General test for the initial text preprocessing function.
        """
        examples = [
            "SQL Server - any equivalent of Excel's CHOOSE function?",
            "How to free c++ memory vector<int> * arr?",
        ]
        answers = [
            "sql server equivalent excels choose function",
            "free c++ memory vectorint arr",
        ]
        for ex, ans in zip(examples, answers):
            self.assertEqual(text_prepare(ex), ans)


if __name__ == "__main__":
    unittest.main()
