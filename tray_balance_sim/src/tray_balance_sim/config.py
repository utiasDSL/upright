import yaml


class Config:
    def __init__(self, path):
        with open(path) as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, keys):
        data = self.data
        for key in keys:
            data = data[key]
        return data


if __name__ == "__main__":
    import IPython

    config = Config("test.yaml")
    config["bar", "bax"]

    IPython.embed()
