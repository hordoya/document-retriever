import unittest
from retriever import Retriever

class TestRetriever(unittest.TestCase):
    def test_query_returns_correct_chunk(self):
        r = Retriever()
        r.add_documents(["documents/example.txt"])
        results = r.query("What is the capital of Italy?")
        self.assertTrue(any("Rome" in chunk for chunk in results))

if __name__ == "__main__":
    unittest.main()
