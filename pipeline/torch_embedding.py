import pandas
from torch.nn.functional import one_hot
import torchtext
import torch, torch.nn, torch.nn.functional, torch.utils.data, torch.optim
import csv

# NOTE: In order to usee this, the train file must have been changed to use NUMERICAL classes - just using the stock classes won't work
# If you've loaded the csv in with pandas, you can do  df['Y'] = df['Y'].map({'HQ': 0, "LQ_CLOSE": 1, "LQ_EDIT": 2})
DEFAULT_TRAIN_FILE = "3_fold/nn_train_0.csv"
DEFAULT_TEST_FILE = "3_fold/nn_test_0.csv"
DEFAULT_OUT_FILE = "out.csv"
NUM_CLASSES = 3
HIDDEN_SIZE = 32
BATCH_SIZE = 5
NUM_EPOCHS = 1
NUM_GRU_LAYERS = 1
LEARNING_RATE = 0.002


class Classifier(torch.nn.Module):
    def __init__(self, body_field: torchtext.data.Field, num_classes: int):
        super().__init__()
        self.vocab_size, self.embed_dim = body_field.vocab.vectors.shape
        print(self.vocab_size, self.embed_dim)
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(body_field.vocab.vectors)
        self.gru = torch.nn.GRU(
            self.embed_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_GRU_LAYERS,
            batch_first=True,
        )
        self.fully_connected = torch.nn.Linear(HIDDEN_SIZE, num_classes)

    def forward(self, data, h):
        res = self.embedding(data)
        res, h = self.gru(res, h)
        res = torch.nn.functional.relu(torch.mean(res, dim=-2))
        res = self.fully_connected(res)

        return res, h

    def get_initial_hidden(self):
        return torch.zeros(NUM_GRU_LAYERS, BATCH_SIZE, HIDDEN_SIZE).cuda()


def get_file_data(filename: str):
    all_data = pandas.read_csv(filename)
    return all_data[["BodyCleaned", "Y"]][:50]


def main(train_file: str, test_file: str, out_file: str):
    body_field = torchtext.data.Field(sequential=True, lower=True)
    label_field = torchtext.data.Field(sequential=False, use_vocab=False)
    num_field = torchtext.data.Field(sequential=False, use_vocab=False)
    train_set, test_set = torchtext.data.TabularDataset.splits(
        path="./",
        train=train_file,
        test=test_file,
        format="csv",
        fields=[("num", num_field), ("BodyCleaned", body_field), ("Y", label_field)],
        skip_header=True,
    )
    body_field.build_vocab(train_set, vectors="fasttext.en.300d", min_freq=20)
    label_field.build_vocab()
    num_field.build_vocab()
    net = Classifier(body_field, NUM_CLASSES).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()
    for _ in range(NUM_EPOCHS):
        for i, item in enumerate(
            torchtext.data.Iterator(train_set, batch_size=BATCH_SIZE)
        ):
            if i % 100 == 0:
                print(i)
            hidden = net.get_initial_hidden()
            data, target = item.BodyCleaned.cuda(), item.Y.cuda()
            # Throw anything away less than the batch size
            if len(data) == 0 or data.shape[1] < BATCH_SIZE:
                continue

            optimizer.zero_grad()
            model_output, _ = net(data.T, hidden)
            loss = loss_function(model_output[: data.shape[1]], target)
            loss.backward()
            optimizer.step()

    print("Testing...")
    num_correct = 0
    predictions = {}
    for item in torchtext.data.Iterator(test_set, batch_size=BATCH_SIZE):
        data, target = item.BodyCleaned.cuda(), item.Y.cuda()
        # Throw anything away less than the batch size
        if data.size()[1] < BATCH_SIZE:
            continue

        with torch.no_grad():
            hidden = net.get_initial_hidden()
            model_out, _ = net(data.T, hidden)
            prediction = model_out.argmax(dim=1, keepdim=True)
            for i, sample_num in enumerate(item.num):
                predictions[sample_num.item()] = (
                    prediction[i].item(),
                    item.Y[i].item(),
                )
            correct = prediction.eq(target.view_as(prediction)).sum().item()
            num_correct += correct

    print("Accuracy: ", num_correct / len(test_set))
    with open(out_file, "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["num", "predicted", "Y"])
        for num, (predicted, Y) in sorted(predictions.items(), key=lambda x: x[0]):
            writer.writerow([num, predicted, Y])


if __name__ == "__main__":
    main(DEFAULT_TRAIN_FILE, DEFAULT_TEST_FILE, DEFAULT_OUT_FILE)
