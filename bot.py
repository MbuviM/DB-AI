import os
from openai import OpenAI
from peewee import Model, MySQLDatabase, TextField, SQL
from tidb_vector.peewee import VectorField
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Init OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embedding_model = "text-embedding-3-small"
embedding_dimensions = 1536

# Init TiDB connection
db = MySQLDatabase(
    'DATA',
    user=os.getenv('TIDB_USERNAME'),
    password=os.getenv('TIDB_PASSWORD'),
    host=os.getenv('TIDB_HOST'),
    port=4000,
    ssl_verify_cert=True,
    ssl_verify_identity=True
)

# Read the document and parse questions and answers
with open("data.txt", "r", encoding="utf-8") as file:
    qa_pairs = []
    for line in file.readlines():
        if line.startswith("Q:"):
            question = line[3:].strip()
        elif line.startswith("A:"):
            answer = line[3:].strip()
            qa_pairs.append((question, answer))

# Define a model with a VectorField to store the embeddings
class DocModel(Model):
    question = TextField()
    answer = TextField()
    embedding = VectorField(dimensions=embedding_dimensions)

    class Meta:
        database = db
        table_name = "qa_embedding_test"
    
    def __str__(self):
        return f"Q: {self.question}\nA: {self.answer}"

db.connect()
db.drop_tables([DocModel])
db.create_tables([DocModel])

# Insert the QA pairs and their embeddings into TiDB
questions = [pair[0] for pair in qa_pairs]
answers = [pair[1] for pair in qa_pairs]
embeddings = [
    r.embedding
    for r in client.embeddings.create(
      input=questions, model=embedding_model
    ).data
]

data_source = [
    {"question": q, "answer": a, "embedding": emb}
    for q, a, emb in zip(questions, answers, embeddings)
]
DocModel.insert_many(data_source).execute()

# Query the most similar document to the question and return the answer
def get_answer(question):
    question_embedding = client.embeddings.create(input=question, model=embedding_model).data[0].embedding
    related_doc = DocModel.select(
        DocModel.answer, DocModel.embedding.cosine_distance(question_embedding).alias("distance")
    ).order_by(SQL("distance")).first()

    if related_doc:
        return related_doc.answer
    else:
        return "Sorry, I couldn't find an answer to that question."

# Example usage
question = "Neonatal Intensive Care Unit (NICU)?"
answer = get_answer(question)

print(f"Question: {question}")
print(f"Answer: {answer}")

db.close()
