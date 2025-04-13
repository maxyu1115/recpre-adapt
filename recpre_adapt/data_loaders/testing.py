from recpre_adapt.data_loaders import PoorMansDataLoaderBase

class TestingDataLoaderWrapper(PoorMansDataLoaderBase):
    """
    A dataloader that returns a fixed batch of data.
    This is for TESTING ONLY!!!
    """
    def __init__(self, actual_dataloader: PoorMansDataLoaderBase):
        self.x = actual_dataloader.get_batch("train")

    def get_batch(self, split: str):
        return self.x
