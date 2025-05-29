import torch
from torch.utils.data import DataLoader, TensorDataset
from train.trainer import train_one_epoch, validate


def test_train_and_validate():
    # Dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return {"out": self.conv(x)}

    model = DummyModel()
    dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.randint(0, 3, (4, 32, 32)))
    loader = DataLoader(dataset, batch_size=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    # Train
    time, loss = train_one_epoch(
        model, loader, criterion, optimizer, torch.device("cpu"), scaler, epoch=1
    )
    assert isinstance(time, float)
    assert isinstance(loss, float)
    # Validate
    acc, miou, val_loss = validate(
        model, loader, criterion, torch.device("cpu"), num_classes=3, epoch=1
    )
    assert 0 <= acc <= 1
    assert 0 <= miou <= 1
