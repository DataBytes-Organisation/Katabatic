from katabatic_spi import KatabaticSPI

class MockDataGen2(KatabaticSPI):
    def load_data(self, data_pathname):
        print("MockDataGen2.load_data()")

    def split_data(self, data_frame, train_test_ratio):
        print("MockDataGen2.split_data()")

    def fit_model(self, data_frame):
        print("MockDataGen2.fit_model()")
