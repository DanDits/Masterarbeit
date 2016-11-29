import matplotlib.pyplot as plt
import pickle


def load_and_show_fig(file_name):
    with open(file_name, 'rb') as data:
        # noinspection PyUnusedLocal
        ax = pickle.load(data)
    plt.show()


def save_fig(ax_data, file_name):
    with open(file_name,'wb') as fid:
        pickle.dump(ax_data, fid)
