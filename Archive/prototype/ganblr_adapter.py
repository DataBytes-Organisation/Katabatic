# pip install ganblr

from katabatic_spi import KatabaticSPI

class GanblrAdapter(KatabaticSPI):
    def load_data(self, data_pathname):
        print("GanblrAdapter.load_data()")

    def split_data(self, data_frame, train_test_ratio):
        print("GanblrAdapter.split_data()")

    def fit_model(self, data_frame):
        print("GanblrAdapter.fit_model()")
