import torch
from torch.utils import data
import json

class DeepmoonDataset(data.Dataset):

    def __init__(self, json_path):

        def moves2tensor(json_moves):
            grid = torch.zeros(3, 18, 11)
            for move in json_moves:
                c = 0 if move["IsStart"] else 2 if move["IsEnd"] else 1
                h = int(move["Description"][1:]) - 1
                w = ord(move["Description"][0].lower()) - 97
                grid[c, h, w] = 1
            return grid

        with open(json_path) as json_file:
            json_data = json.load(json_file)

        self.moves = [moves2tensor(moonboard["Moves"]) for moonboard in json_data]
        self.labels = [moonboard["Grade"] for moonboard in json_data]
        self.grades = ['5+', '6A', '6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+',
                       '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, item):
        return {'moves': self.moves[item], 'label': self.grades.index(self.labels[item])}


if __name__ == '__main__':
    import sys
    json_path = sys.argv[1]
    dataset = DeepmoonDataset(json_path)
    print(len(dataset))