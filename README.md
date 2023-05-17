# ImageSearch

## Purpose
Given an input, finds similar image using cosine similarity.

## How it was built
The data was turned into a feature embedding vector through a library called `img2vec`. Then, the vectors were pushed into a vector database (namely pinecone). Then, using pinecone, we query for the most similar match using cosine similarity.

## How to use it
* Create a new pinecone index.
* Add images to the `imgs/` folder
* Make sure to set the environment variable `PINECONE_APIKEY` to your pinecone API token.
* Run `updateDB.py`
* Run `main.py`. Update the code to search similarities for an image of your choice.