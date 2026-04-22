import torch

class HFF:
    @staticmethod
    def fuse(branches):
        out = branches[0]
        outputs = [out]

        for i in range(1, len(branches)):
            out = branches[i] + out
            outputs.append(out)

        return torch.cat(outputs, dim=1)
