import cv2

from miga_emblaze.emblaze import EmbeddingSet, Embedding, Field
import os
import numpy as np


def to_EmbeddingSet(embeddings, colors):
    emb_set = EmbeddingSet([Embedding({Field.POSITION: embeddings, Field.COLOR: colors})])
    return emb_set


def to_Embedding(embeddings, colors):
    return Embedding({Field.POSITION: embeddings, Field.COLOR: colors})


def add_new_embedding(sample_idx_list, embedding_modality, trans, old_embeddings, color_array, old_img_list):
    need = ['img', 'graph']
    embedding_dir = '/rhome/lianyu.zhou/cache/cluster_test_set_gformer_l3_b256_atom60'
    from miga_emblaze.emblaze import ImageThumbnails
    from miga_emblaze.emblaze import EmbeddingSet, Embedding
    from miga_emblaze.emblaze.utils import Field

    new_img_list = []
    for idx in sample_idx_list:
        modal_imgs_dir = f'/rhome/lianyu.zhou/dataset/omics_{need[embedding_modality]}_vis'
        read_path = os.path.join(modal_imgs_dir, f"{idx:05d}.png")
        img = cv2.imread(read_path)[..., ::-1]
        new_img_list.append(img)
    embeddings = np.load(os.path.join(embedding_dir, f"{need[embedding_modality]}_embedding.npy"))
    new_embeddings = embeddings[sample_idx_list]
    transformed_embeddings = trans.transform(new_embeddings)
    merge_old_embeddings = np.concatenate([old_embeddings, transformed_embeddings], axis=0)
    new_color_array = np.concatenate([color_array, np.ones((len(sample_idx_list),))], axis=0)
    new_reduced = EmbeddingSet([Embedding({Field.POSITION: merge_old_embeddings, Field.COLOR: new_color_array})])
    new_reduced.compute_neighbors(metric='euclidean', n_neighbors=100)
    new_img_list.extend(old_img_list)
    thumbnails = ImageThumbnails(new_img_list)
    return new_reduced, thumbnails
