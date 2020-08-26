# Load the confusion matrix of the output

import numpy as np


class ConfusionMatrix:
    """
    The definition of confusion matrix here is the same as the definition of the sklearn.metrics:
    A confusion matrix C is such that C[i, j] is equal to the number of observations known to be in group i and
    predicted to be in group j.
    """

    def __init__(self, load_path):
        """

        Args:
            load_path: str.
        """
        self._cfn_mtx = np.load(load_path)

        height, width = self._cfn_mtx.shape
        assert height == width
        self.num_class = height

    def get_submatrix(self, indices, to_probability=False, use_log=False):
        """
        Gets the confusion matrix that only considers the sub-indices

        Args:
            indices: List[int]
            to_probability: bool = False. If True, the return confusion matrix will be normalized with probability.
            use_log: bool=False. If True, the return confusion matrix will be in log value. (Only will work if
                to_probability is True)
        """
        num_indices = len(indices)
        if num_indices == 0: return []
        if num_indices > self.num_class:
            raise ValueError("The number of indices is greater than the number of classes in the confusion matrix!")
        for i in indices:
            if i < 0 or i >= self.num_class:
                raise ValueError("Invalid index!", i)

        sub_mtx = self._cfn_mtx[np.ix_(indices, indices)]
        if to_probability:
            sub_mtx = self._normalize_to_probability(sub_mtx)
            if use_log:
                sub_mtx = np.log(sub_mtx)
        return sub_mtx

    def __str__(self):
        return str(self._cfn_mtx)

    def __len__(self):
        return self.num_class

    def __getitem__(self, item):
        return self._cfn_mtx[item]

    def _normalize_to_probability(self, mtx):
        """ Normalizes the confusion matrix to probability """

        # Need to change the np.sum(mtx, axis=1) shape to (-1, 1) otherwise the division is wrong.
        return mtx / np.sum(mtx, axis=1)[:, np.newaxis]


if __name__ == '__main__':
    cfn_mtx = ConfusionMatrix(
        "/home/qinru/avl/playground/vision_semantic_segmentation/external_data/confusion_matrix/run_trad_cnn/cfn_mtx.npy")

    print(cfn_mtx)
    print(len(cfn_mtx))

    # indices = [1, 3, 5, 6]
    indices = [2, 1, 8, 10, 3]
    sub_mtx = cfn_mtx.get_submatrix(indices, True)
    # for i in range(len(indices)):
    #     assert sub_mtx[i, 0] == cfn_mtx[indices[i], indices[0]]
    print(sub_mtx)

    # Test if the sum of each class is correct
    map = np.zeros((300, 400, 5))
    mask = np.zeros((300, 400)).astype(bool)
    mask[:, 0] = True

    map[mask, :] += sub_mtx[0, :].reshape(1, -1)

    print()
