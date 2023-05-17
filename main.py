import os

from PIL import Image
from img2vec_pytorch import Img2Vec

import pinecone

PINECONE_APIKEY = os.environ['PINECONE_APIKEY']
pinecone.init(api_key=PINECONE_APIKEY, environment="asia-northeast1-gcp")
index = pinecone.Index("imagesearch")

img2vec = Img2Vec(cuda=False)

def search_image(img, top_k=2):
    path = os.path.join(os.getcwd(), "imgs", img)
    image = Image.open(path)
    vector = img2vec.get_vec(image).tolist()
    res = index.query(vector=vector, top_k=top_k)
    matches = res['matches']
    out = []
    for match in matches:
        out.append([match['id'], match['score']])
    return out

images = search_image("image.png")
for image in images:
    print(image[0])
