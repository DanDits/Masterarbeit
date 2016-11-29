import matplotlib.pyplot as plt
import pickle


def load_and_show_fig(file_name):
    """
    Loads and shows the figure saved in the pickled file.
    Uses pyplot.show(), so this will block the current thread.
    :param file_name: The path to the file to open.
    :return: None
    """
    with open(file_name, 'rb') as data:
        # noinspection PyUnusedLocal
        ax = pickle.load(data)
    plt.show()


def save_fig(ax_data, file_name):
    """
    Saves the given axes data to the given file using pickle. Can be restored
    later by using load_and_show_fig(file_name).
    :param ax_data: The axes data obtained by plt.axes(), use before showing figure!
    :param file_name: The file name to save the pickle dump to.
    :return: None
    """
    with open(file_name,'wb') as fid:
        pickle.dump(ax_data, fid)
