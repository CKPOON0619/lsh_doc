

from fastapi.testclient import TestClient
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import nest_asyncio
from .context import app
import unittest


RandomState(MT19937(SeedSequence(123456789)))
client = TestClient(app)
nest_asyncio.apply()


class TestApp(unittest.TestCase):
    def test_response(self):
        response1 = client.post("/register/items", json={
            "id": ["111", "222", "333"],
            "type": "dimension",
            "name": ["real prop1", "testing prop2", "testing prop3"],
            "description": ["this is a real prop1.", "this is a testing prop2", "this is a testing prop3"]})
        self.assertEqual(
            np.array(response1.json()["embeddings"]).shape, (3, 512))

        q_embd = response1.json()["embeddings"][1]
        response2 = client.get(
            "/query/items", json={"type": "dimension", "content": "description", "embeddings": q_embd})
        self.assertTrue('222' in response2.json())

        response3 = client.get("/query/tables", json={"type": "dimension"})
        self.assertTrue('name' in list(response3.json().keys()))
        self.assertTrue('description' in response3.json().keys())


if __name__ == '__main__':
    unittest.main()
