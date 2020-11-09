import pandas as pd
import numpy as np
import csv

# define files
# Just change the location if it changes
file_name_train = "/Users/miranda/PycharmProjects/pythonProject6/train.csv"
file_name_test = "/Users/miranda/PycharmProjects/pythonProject6/valid.csv"

#read files
df_train = pd.read_csv(file_name_train)
df_test = pd.read_csv(file_name_test)

#define lists
train_list = []
test_list = []

#append to lists
for i in df_train["Id"]:
    train_list.append(i)

for x in df_test["Id"]:
    test_list.append(x)


#Return lists of 100 Train Ids
def train_list_return():
    print("Train")
    chunks = [train_list[x:x+100] for x in range(0, len(train_list), 100)]
    i=0
    for small_train_list in chunks:
        i+=1
        print(i)
        print(small_train_list)

train_list_return()

#Return lists of 100 Test Ids
def test_list_return():
    print("Test")
    chunks = [test_list[x:x+100] for x in range(0, len(test_list), 100)]
    i=0
    for small_test_list in chunks:
        i+=1
        print(i)
        print(small_test_list)

test_list_return()
