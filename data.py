from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import PIL_to_tensor


def get_dataloader(
    path, split="train", resize=None, image_column="image", column_filter=None, batch_size=32, shuffle=True
):
    """Creates a DataLoader for a dataset located at the specified path.

    Uses the Hugging Face Datasets library to load a dataset, apply specified transformations, and create a DataLoader object. Intended to be used with datasets with the "image" column containing PIL images.

    Args:
        path (str): The path to the dataset. Either a local or remote (Hugging Face Hub) path.
        split (str, optional): The dataset split to use (e.g., "train", "test"). Defaults to "train". Can also use parts of the dataset (e.g., "train[:10%]", "train[10:20]").
        resize (tuple, optional): The size to which images should be resized (width, height). Defaults to None.
        image_column (str, optional): The name of the column containing the images. Defaults to "image".
        column_filter (tuple, optional): A tuple containing the column name and value to filter the dataset. Defaults to None.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        DataLoader: A DataLoader object for the specified dataset.
    """
    dataset = load_dataset(path, split=split)
    dataset = dataset.map(
        lambda e: PIL_to_tensor(e, resize=resize, image_column=image_column),
        remove_columns=[image_column],
        batched=True,
        load_from_cache_file=False,
    )
    dataset.set_format(type="torch", columns=["pixel_values"])

    if column_filter is not None:
        dataset = dataset.filter(lambda e: e[column_filter[0]] == column_filter[1])

    return DataLoader(
        dataset["pixel_values"],
        batch_size=batch_size,
        shuffle=shuffle,
    )
