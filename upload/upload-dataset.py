from griptape.drivers import PineconeVectorStoreDriver, OpenAiEmbeddingDriver, PongoRerankDriver
from datasets import load_dataset
from dotenv import load_dotenv
import os
import pongo
from griptape.artifacts import TextArtifact
import uuid
load_dotenv()

embedding_driver = OpenAiEmbeddingDriver(model="text-embedding-3-small")

vector_store_driver = PineconeVectorStoreDriver(
    api_key=os.environ["PINECONE_API_KEY"],
    environment='us-west-2',
    index_name='hotpot-questions',
    embedding_driver=embedding_driver,
)

artifacts = []

dataset = load_dataset("hotpot_qa", 'distractor', split='validation')

for datapoint in dataset:
    artifacts.append(TextArtifact(f"Question: {datapoint['question']}\nAnswer: {datapoint['answer']}"))


vector_store_driver.upsert_text_artifacts({"griptape": artifacts})

