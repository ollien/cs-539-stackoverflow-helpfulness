import pandas
from torch.nn.functional import one_hot
import torchtext
import torch, torch.nn, torch.nn.functional, torch.utils.data, torch.optim

TRAIN_FILE = "../train-trunc.csv"
NUM_CLASSES = 3


class Classifier(torch.nn.Module):
    def __init__(self, body_field: torchtext.data.Field, num_classes: int):
        super().__init__()
        vocab_size, embed_dim = body_field.vocab.vectors.shape
        print(vocab_size, embed_dim)
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(body_field.vocab.vectors)
        self.fully_connected = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, data):
        res = self.embedding(data)
        res = self.fully_connected(res)

        return res


def get_file_data(filename: str):
    all_data = pandas.read_csv(filename)
    return all_data[["BodyCleaned", "Y"]][:50]


def main():
    body_field = torchtext.data.Field(sequential=True, lower=True)
    label_field = torchtext.data.Field(sequential=False, use_vocab=False)
    dataset = torchtext.data.TabularDataset(
        path=TRAIN_FILE,
        format="csv",
        fields=[("BodyCleaned", body_field), ("Y", label_field)],
        skip_header=True,
    )
    body_field.build_vocab(dataset, vectors="glove.6B.100d", min_freq=20)
    label_field.build_vocab()
    net = Classifier(body_field, NUM_CLASSES).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    loss_function = torch.nn.CrossEntropyLoss()
    train_set, test_set = dataset.split(0.7)
    for _ in range(10):
        for i, item in enumerate(torchtext.data.Iterator(train_set, batch_size=100)):
            if i % 100 == 0:
                print(i)
            data, target = item.BodyCleaned.cuda(), item.Y.cuda()
            optimizer.zero_grad()
            model_output = net(data.T)
            loss = loss_function(model_output, target)
            loss.backward()
            optimizer.step()

    print("Testing...")
    num_correct = 0
    for item in torchtext.data.Iterator(test_set, batch_size=100):
        data, target = item.BodyCleaned.cuda(), item.Y.cuda()
        with torch.no_grad():
            model_out = net(data.T)
            prediction = model_out.argmax(dim=1, keepdim=True)
            correct = prediction.eq(target.view_as(prediction)).sum().item()
            num_correct += correct

    print("Accuracy: ", num_correct / len(test_set))


if __name__ == "__main__":
    main()
