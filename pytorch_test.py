import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

def fetch(url):
	import requests, gzip, os, hashlib, numpy
	fp = os.path.join("./tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
	if os.path.isfile(fp):
		with open(fp, "rb") as f:
			dat = f.read()
	else:
		with open(fp, "wb") as f:
			dat = requests.get(url).content
			f.write(dat)
	return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")


class BobNet(torch.nn.Module):
	def __init__(self):
		super(BobNet, self).__init__()
		self.l1 = nn.Linear(784, 128)
		self.act = nn.ReLU()
		self.l2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.l1(x)
		x = self.act(x)
		x = self.l2(x)
		return x

model = BobNet()
model(torch.tensor(X_train[0:10].reshape((-1,28*28))).float())

BS=12

loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

for i in (t := trange(1000)):

