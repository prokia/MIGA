import torch
from torch_geometric.data import Data, Batch

from utils.geometric_graph import to_dense, fetch_geometric_batch
from utils.tensor_operator import tensor2array
from .default_collate import default_collate
# from ..creat_dataloader import collate_fn_register

# @collate_fn_register
class BatchDenseMatrix(Data):
    def __init__(self, batch=None, **kwargs):
        super(BatchDenseMatrix, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(items, transforms=None, has_img=True):

        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        re_dict = {}

        data_list = list(item.pop('graphs') for item in items)
        geometric_batch = Batch().from_data_list(data_list)
        matrix_graphs, node_masks = to_dense(*fetch_geometric_batch(geometric_batch, ['edge_attr', 'batch']))
        matrix_graphs = matrix_graphs.to(None, 'float')
        re_dict.update({'graphs': matrix_graphs, 'node_masks': node_masks})

        if has_img:
            img_collat = default_collate(items)
            def flatten_first(f_key):
                img_p = img_collat[f_key]
                b, n, c, h, w = img_p.size()
                img_collat[f_key] = torch.reshape(img_p, (b * n, c, h, w))
            for key_name in ['imgs_ins']:
                flatten_first(key_name)
            re_dict.update(img_collat)

        if hasattr(data_list[0], 'y') and data_list[0].y is not None:
            re_dict['y'] = torch.cat([d.y for d in data_list])
        return re_dict

# @collate_fn_register
class BatchMaskingGraph(Data):

    def __init__(self, batch=None, **kwargs):
        super(BatchMaskingGraph, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(items, transforms=None):

        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        if isinstance(items[0], tuple):
            has_img = True
            data_list = [item[0] for item in items]
            img_list = [item[1] for item in items]
        else:
            has_img = False
            data_list = items
            img_list = items

        if transforms is not None:
            if not isinstance(transforms, list):
                transforms = [transforms]
            for transform in transforms:
                data_list = [transform(d) for d in data_list]

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMaskingGraph()

        # print(data_list[0])
        # keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context", 'x', 'edge_attr', 'edge_index', "mask_node_idx", 'mask_node_label', 'mask_edge_idx', 'mask_edge_label']
        # keys = ['edge_index_substruct', 'x', 'edge_index', 'edge_attr_substruct', 'x_context', 'overlap_context_substruct_idx', 'mask_node_label', 'edge_attr', 'masked_atom_indices', 'center_substruct_idx', 'x_substruct', 'edge_attr_context', 'masked_x', 'edge_index_context']
        # print(keys)
        for key in keys:
            batch[key] = []
        batch.batch = []
        # used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_node = 0
        cumsum_edge = 0
        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        imgs = []
        for data, img in zip(data_list, img_list):
            # If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
                batch.batch_overlapped_context.append(
                    torch.full((len(data.overlap_context_substruct_idx),), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ##batching for the main graph
                for key in data.keys:
                    # for key in ['x', 'edge_attr', 'edge_index']:

                    # if not "context" in key and not "substruct" in key:
                    if key in ['x', 'edge_attr', 'edge_index']:
                        item = data[key]
                        # item = item + cumsum_main if batch.cumsum(key, item) else item
                        if key in ['edge_index']:
                            item = item + cumsum_main
                        batch[key].append(item)

                    ###batching for the substructure graph
                    elif key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                        item = data[key]
                        item = item + cumsum_substruct if batch.cumsum(key, item) else item
                        batch[key].append(item)


                    ###batching for the context graph
                    elif key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context",
                                 "x_context"]:
                        item = data[key]
                        item = item + cumsum_context if batch.cumsum(key, item) else item
                        batch[key].append(item)

                    elif key in ['masked_atom_indices']:
                        item = data[key]
                        item = item + cumsum_node
                        batch[key].append(item)

                    elif key == 'connected_edge_indices':
                        item = data[key]
                        item = item + cumsum_edge
                        batch[key].append(item)

                    else:
                        item = data[key]
                        batch[key].append(item)

                cumsum_node += num_nodes
                cumsum_edge += data.edge_index.shape[1]
                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

                imgs.append(img)

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        if has_img:
            imgs_batch = torch.stack(imgs, 0)
            return batch.contiguous(), imgs_batch
        else:
            return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx",
                       "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(items, transform=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        data_list = list(item.pop('graphs') for item in items)
        # b = len(data_list)
        img_collat = default_collate(items)

        def flatten_first(f_key):
            img_p = img_collat[f_key]
            b, n, c, h, w = img_p.size()
            img_collat[f_key] = torch.reshape(img_p, (b * n, c, h, w))

        for key_name in ['imgs_ins']:
            flatten_first(key_name)

        if transform is not None:
            data_list = [transform(d) for d in data_list]

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)

        img_collat['graphs'] = batch
        return img_collat

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


# @collate_fn_register
class BatchAE(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0


# @collate_fn_register
class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(items, transform=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        # keys = [set(data.keys) for data in data_list]
        # keys = list(set.union(*keys))
        # assert 'batch' not in keys
        if isinstance(items[0], tuple):
            has_img = True
            data_list = [item[0] for item in items]
            img_list = [item[1] for item in items]
        else:
            has_img = False
            data_list = items
            img_list = items

        if transform is not None:
            data_list = [transform(d) for d in data_list]

        batch = BatchSubstructContext()

        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct",
                "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context", 'x',
                'edge_attr', 'edge_index']

        for key in keys:
            batch[key] = []

        batch.batch = []
        # used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        imgs = []
        for data, img in zip(data_list, img_list):
            # If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
                batch.batch_overlapped_context.append(
                    torch.full((len(data.overlap_context_substruct_idx),), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ##batching for the main graph
                # for key in data.keys:
                for key in ['x', 'edge_attr', 'edge_index']:
                    if not "context" in key and not "substruct" in key:
                        item = data[key]
                        # item = item + cumsum_main if batch.cumsum(key, item) else item
                        if key in ['edge_index']:
                            item = item + cumsum_main
                        batch[key].append(item)

                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

                imgs.append(img)

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        if has_img:
            imgs_batch = torch.stack(imgs, 0)
            return batch.contiguous(), imgs_batch
        else:
            return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx",
                       "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


# @collate_fn_register
class BatchSubstructContext3D(Data):
    """A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier. """

    ''' Specialized batching for substructure context pair! '''

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(items, transform=None):
        """Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""

        data_list = [item[0] for item in items]
        imgs = torch.stack([item[1] for item in items], 0)

        if transform is not None:
            data_list = [transform(d) for d in data_list]

        batch = BatchSubstructContext()
        keys = [
            'center_substruct_idx', 'edge_attr_substruct',
            'edge_index_substruct', 'x_substruct', 'overlap_context_substruct_idx',
            'edge_attr_context', 'edge_index_context', 'x_context',
            'positions', 'x', 'edge_attr', 'edge_index'
        ]

        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0

        for data in data_list:
            if hasattr(data, 'x_context'):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
                batch.batch_overlapped_context.append(
                    torch.full((len(data.overlap_context_substruct_idx),), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                # batching for the main graph
                for key in ['x', 'edge_attr', 'edge_index', 'positions']:
                    item = data[key]
                    if key in ['edge_index']:
                        item = item + cumsum_main
                    batch[key].append(item)

                # batching for the substructure graph
                for key in ['center_substruct_idx', 'edge_attr_substruct',
                            'edge_index_substruct', 'x_substruct']:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                # batching for the context graph
                for key in ['overlap_context_substruct_idx', 'edge_attr_context',
                            'edge_index_context', 'x_context']:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

            else:
                raise Exception

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.__cat_dim__(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous(), imgs

    def __cat_dim__(self, key):
        return -1 if key in ['edge_index', 'edge_index_substruct', 'edge_index_context'] else 0

    def cumsum(self, key, item):
        """If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute. """
        return key in ['edge_index', 'edge_index_substruct',
                       'edge_index_context',
                       'overlap_context_substruct_idx',
                       'center_substruct_idx']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
