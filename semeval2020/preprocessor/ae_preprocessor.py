import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from semeval2020.factory_hub import preprocessor_factory, abstract_preprocessor


class AutoEncoder(nn.Module):

    def __init__(self, input_size, learning_rate=1e-3, weight_decay=1e-5, num_epochs=128, batch_size=128):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 20))
        self.decoder = nn.Sequential(
            nn.Linear(20, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, input_size))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_size = input_size

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(self.num_epochs):
            loss = None
            for data in dataloader:
                data = Variable(data)
                # ===================forward=====================
                output = self(data)
                loss = criterion(output, data)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            # print(f"epoch [{epoch + 1}/{self.num_epochs}], loss:{loss.item():.4f}")

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def transform(self, dataset):
        result = []
        for data in dataset:
            result.append(self.encoder(Variable(torch.Tensor(data))).detach().numpy())
        return result


preprocessor_factory.register("AutoEncoder", AutoEncoder)
