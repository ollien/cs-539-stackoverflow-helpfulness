import numpy as np
import pickle

with open('bodyVectorized.pkl', 'rb') as f:
	bodyVectorized = pickle.load(f)#list of tensors
#bodyVectorized = [np.array([[1,2,3],[4,5,6]]), np.array([[1,2,3],[4,5,6],[7,8,9]])]
largestSentence = bodyVectorized[0]
largestLength = largestSentence.shape[0]
for x in bodyVectorized:
	if x.shape[0] > largestLength:
		largestLength = x.shape[0]
		largestSentence = x
toDump = []
for x in bodyVectorized:
	toAppend = x
	padding = np.zeros((largestLength - x.shape[0], x.shape[1]))
	toAppend = np.concatenate((padding, toAppend))
	toDump.append(toAppend)
	#print(toAppend)

pickle.dump(toDump, 'paddedVectorizedBody.pkl')