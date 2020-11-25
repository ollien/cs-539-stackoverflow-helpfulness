import pandas as pd
from torch.nn.functional import one_hot
import torchtext
import torch, torch.nn, torch.nn.functional, torch.utils.data, torch.optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

# NOTE: In order to usee this, the train file must have been changed to use NUMERICAL classes - just using the stock classes won't work
# If you've loaded the csv in with pandas, you can do  df['Y'] = df['Y'].map({'HQ': 0, "LQ_CLOSE": 1, "LQ_EDIT": 2})
TRAIN_FILE = "train-vectorized.csv"
NUM_CLASSES = 3
HIDDEN_SIZE = 32
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_GRU_LAYERS = 1
LEARNING_RATE = 0.002
EMBED_DIM = 1

def labelToVal(x):
	if x == 'LQ_CLOSE':
		return -1
	elif x == 'LQ_EDIT':
		return 0
	else:
		return 1

def parseVector(x):
	masterList = []
	x = x[1:-1]
	x = x.split(',\n')
	for line in x:
		line = line[1:-1]
		line = line.split(',')
		lineArray = []
		for value in line:
			lineArray.append(float(value))
		masterList.append(lineArray)
	return np.array([masterList])

class myDataset(Dataset):
	def __init__(self, csv, isTrain):
		df = pd.read_csv(csv)
		#self.body = pickle.load(bodyNumpyList)#this is a list of numpy arrays that have 0 padding at the beginning
		self.body = df['VectorsToParse']#.apply(lambda x: cleanLine(x))
		if isTrain:
			self.body = self.body[:int(len(self.body) * .7)]
		else:	
			self.body = self.body[int(len(self.body) * .7):]
		self.body.replace('', np.nan, inplace = True)
		self.body.dropna(inplace = True)
		self.label = df['Y'].apply(labelToVal).values.tolist()
		self.body = self.body.apply(parseVector)
		self.length = len(df)
		
	def __len__(self):
		return self.length
	
	def __getitem__(self,index):
		toReturn = [torch.Tensor(self.body[index]), torch.Tensor(np.array([self.label[index]]))]
		return toReturn

class Classifier(torch.nn.Module):
	def __init__(self, embed_dim: int, num_classes: int):
		super().__init__()
		self.gru = torch.nn.GRU(
			embed_dim,
			hidden_size=HIDDEN_SIZE,
			num_layers=NUM_GRU_LAYERS,
			batch_first=True,
		)
		self.fully_connected = torch.nn.Linear(HIDDEN_SIZE, num_classes)

	def forward(self, data, h):
		print(f"data shape: {data.shape}")
		print(f"h shape: {h.shape}")
		res, h = self.gru(data)
		res = torch.nn.functional.relu(torch.mean(res, dim=-2))
		res = self.fully_connected(res)

		return res, h

	def get_initial_hidden(self):
		return torch.zeros(NUM_GRU_LAYERS, BATCH_SIZE, HIDDEN_SIZE)



def main():
	dataSet = myDataset(TRAIN_FILE, True)
	net = Classifier(EMBED_DIM, NUM_CLASSES)
	optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
	loss_function = torch.nn.CrossEntropyLoss()
	for _ in range(NUM_EPOCHS):
		for i, item in enumerate(dataSet):
			if i % 100 == 0:
				print(i)
			hidden = net.get_initial_hidden()
			data, target = item[0], item[1]
			print(f"data shape: {data.shape}")
			print(f"data.T shape: {data.T.shape}")
			print(f"target size: {target.size()}")
			# Throw anything away less than the batch size
			if len(data) == 0 or data.shape[1] < BATCH_SIZE:
				continue

			optimizer.zero_grad()
			model_output, _ = net(data.T, hidden)
			print(model_output)
			print(model_output[: data.shape[1]])
			loss = loss_function(model_output[: data.shape[1]], target)
			loss.backward()
			optimizer.step()

	print("Testing...")
	testSet = myDataset(TRAIN_FILE, False)
	for item in testSet:
		data, target = item[0], item[1]
		# Throw anything away less than the batch size
		if data.size()[1] < BATCH_SIZE:
			continue

		with torch.no_grad():
			hidden = net.get_initial_hidden()
			model_out, _ = net(data.T, hidden)
			prediction = model_out.argmax(dim=1, keepdim=True)
			correct = prediction.eq(target.view_as(prediction)).sum().item()
			num_correct += correct

	print("Accuracy: ", num_correct / len(testSet))


if __name__ == "__main__":
	main()
