from batch import BatchMasking, BatchSubstructContext, BatchSubstructContext3D, BatchMaskingGraph
from torch.utils.data import DataLoader


class DataLoaderMasking(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`) """

    def __init__(self, dataset, transform, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda items: BatchMasking.from_data_list(items, transform),
            **kwargs)


class DataLoaderSubstructContext(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`) """

    def __init__(self, dataset, transform, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list, transform),
            **kwargs)


class DataLoaderSubstructContext3D(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`) """

    def __init__(self, dataset, transform=None, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext3D, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext3D.from_data_list(data_list, transform),
            **kwargs)


class DataLoaderMaskingGraph(DataLoader):
    """Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`) """

    def __init__(self, dataset, transforms, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMaskingGraph, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda items: BatchMaskingGraph.from_data_list(items, transforms),
            **kwargs)
