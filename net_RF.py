def DIL_RF(one_list, ks, length, layer):
    new_list = []
    kk = int((ks-1)/2)
    dist = 2 ** (layer - 1)
    for i in one_list:
        if i < length:
            new_list.append(i)
            for kkk in range(kk):
                if i-dist*(kkk+1) >= 0:
                    new_list.append(i-dist*(kkk+1))
                if i+dist*(kkk+1) < length:
                    new_list.append(i+dist * (kkk + 1))
    new_list = list(set(new_list))
    new_list.sort()
    return new_list


def receptive_field(seq_length, kernel_size, layer):
    for one_point in [int(seq_length / 2)]:
        print('position:', one_point)
        for L in range(1, layer + 1):
            RF = [one_point]
            for LL in range(L, 0, -1):
                RF = DIL_RF(RF, kernel_size, seq_length, LL)
            print('layer:{:d} \t distance:{:d} \t size:{:d}'.format(L, RF[-1] - RF[0] + 1, len(RF)))
        print()