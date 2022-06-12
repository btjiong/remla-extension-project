"""
    Contains tests concering the overall quality of the trained model.
"""

import unittest

import joblib

from so_classifier.text_preprocessing import text_prepare


class TestTextPrepare(unittest.TestCase):
    """
    Contains tests concering the overall quality of the trained model.
    """

    def test_simple_noise_robustness(self):
        """
        General test to see if the model  ignores noise.
        """

        tfidf_model = joblib.load("output/tfidf_model.joblib")
        tfidf_vectorizer = joblib.load("output/tfidf_vectorizer.joblib")
        mlb = joblib.load("output/mlb.joblib")


        title = "Python Script execute commands in Terminal"
        prepared_title = text_prepare(title)
        vectorized_title = tfidf_vectorizer.transform([prepared_title])
        prediction = tfidf_model.predict(vectorized_title)
        prediction = mlb.inverse_transform(prediction)[0]

        noise_title = "From a Python Script execute commands in a Terminal"
        prepared_title_noise = text_prepare(noise_title)
        vectorized_titl_noise = tfidf_vectorizer.transform([prepared_title_noise])
        prediction_noise = tfidf_model.predict(vectorized_titl_noise)
        prediction_noise = mlb.inverse_transform(prediction_noise)[0]


        self.assertEquals(prediction, prediction_noise)

if __name__ == "__main__":
    unittest.main()
