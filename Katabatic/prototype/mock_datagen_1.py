from katabatic_spi import KatabaticSPI

class MockDataGen1(KatabaticSPI):
    def load_data(self, data_pathname):
        print("MockDataGen1.load_data()")

    def split_data(self, data_frame, train_test_ratio):
        print("MockDataGen1.split_data()")

    def fit_model(self, data_frame):
        print("MockDataGen1.fit_model()")
