import torch

__all__ = ['order_by_confidence']


def order_by_confidence(skeleton_list):
    n_people = []
    for sk in skeleton_list:  # TODO empty list ( no detection)
        n_people.append(sk.shape[0])

    max_people = max(n_people)
    basic_shape = torch.zeros(2, 3, 25)
    if max_people == 0:
        return [basic_shape for _ in skeleton_list], None

    coef = [torch.einsum('ab->a', x[:, 2, :]) / 25 for x in skeleton_list]
    return_list = []
    idx_list = []
    for c, sk, N in zip(coef, skeleton_list, n_people):
        if N == 0:
            return_list.append(basic_shape)
            idx_list.append([-1])
        elif N == 1:
            tmp = basic_shape.clone()
            tmp[0, ...] = sk
            return_list.append(tmp)
            idx_list.append([0])
        elif N > 1:
            idx = c.argmax()
            c[idx] = -1
            idx2 = c.argmax()
            idx_list.append([idx, idx2])
            return_list.append(torch.stack([sk[idx, ...], sk[idx2, ...]]))

    return return_list, idx_list
