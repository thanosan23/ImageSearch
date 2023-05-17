import os
import glob

from PIL import Image
from img2vec_pytorch import Img2Vec

import pinecone

PINECONE_APIKEY = os.environ['PINECONE_APIKEY']
pinecone.init(api_key=PINECONE_APIKEY, environment="asia-northeast1-gcp")
index = pinecone.Index("imagesearch")

img2vec = Img2Vec(cuda=False)

data = []
for files in glob.glob(os.path.join(os.getcwd(), "imgs", "*")):
    img = Image.open(os.path.join(os.getcwd(), "imgs", files))
    vector = img2vec.get_vec(img).tolist()
    data.append((files, vector))

index.upsert(data)
