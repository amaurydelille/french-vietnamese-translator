from input_embedding import InputEmbeddings

embeddings = InputEmbeddings(5, 5)

embedded = embeddings.forward('Caca qui pue')

print(embedded)