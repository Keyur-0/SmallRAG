from itertools import count
import math
from shlex import join

import ollama
import chromadb
# 1. Load
with open("text.txt") as f:

    text = f.read()  # TODO: read your file

# 2. Chunk
size = 100
overlap = 10
top_k = 3
# text[start:end]

file_length = len(text)
# print(file_length)
chunk_start = 0
chunk_end = size
ids = []
id = 1
chunks = []

# TODO: split with size + overlap
number_of_chunks = math.ceil(1+((file_length -size)/(size-overlap)))

for i in range(number_of_chunks):
    chunks.append(text[chunk_start:chunk_end])
    ids.append(str(id))
    chunk_start += (size -overlap)
    id += 1
    chunk_end += (size -overlap)

#the thing that i did before
# while True:
#     chunks.append(text[chunk_start:chunk_end])
#     ids.append(id)
#     chunk_start += 80
#     id += 1
#     chunk_end += 80
#     print(chunk_end)
#     if chunk_end > file_length:
#         print(chunk_end)
#         chunk_end = file_length
#         if chunk_end == file_length:
#             break
#         ids.append(id)
#         id += 1
#         chunks.append(text[chunk_start:chunk_end])
#         break

# print(chunks)

# 3. Embed + store
response = ollama.embed(
        model = "nomic-embed-text:latest",
        input = chunks
    )
# print(response)

client = chromadb.Client()

# print(client)

collection = client.create_collection(name="Solar_energy")

collection.add(
    ids = ids,
    documents = chunks,
    embeddings = response.embeddings
)
# TODO: embed each chunk, add to vector store

# 4. Query
question = "How do solar panels generate electricity?"
# TODO: embed question, retrieve top-k chunks

embeded_question = ollama.embed(
        model = "nomic-embed-text:latest",
        input = [question]
    )

top_k_chunks = collection.query(
    query_embeddings = embeded_question.embeddings,
    n_results = top_k
)
print(top_k_chunks)

# 5. Augment + Generate
print(question)
# prompt = "Answer using only the supplied context.Answer using only the supplied context.If the context doesn't contain the answer, explicitly say so." + "Question: " + question + "resourse:" + top_k_chunks
docs_list = top_k_chunks.get("documents")

print(docs_list)

for i in docs_list:
    docs = " ".join(i)

print("\n\n\n" + docs)

prompt = "Only answer with this given resourse in this prompt:" +"Question: " + question +"resourse:" + docs
# TODO: build prompt from template + chunks

answer = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    options = {
        "temprature" : 0.3
    }
)

print(answer.message.content)  # TODO: call LLM
