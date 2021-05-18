from pathlib import Path

from vad.vad_utils import parse_vad_label, prediction_to_vad_label, read_label_from_file

# read_label_from_file('data/test_label.txt')
# for i in range(10):
#     test = np.random.randint(0, 2, size=(30,))
#     print(test)
#     print(prediction_to_vad_label(test))
#     print()
data = {}
path = 'data/test_label.txt'
with Path(path).open("r", encoding="utf-8") as f:
    err_num = 0
    for linenum, line in enumerate(f, 1):
        sps = line.strip().split(maxsplit=1)
        if len(sps) == 1:
            print(f'Error happened with path="{path}", id="{sps[0]}", value=""')
        else:
            k, v = sps
        if k in data:
            raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
        # try:
        data[k] = parse_vad_label(v, frame_size=0.032, frame_shift=0.008)
        # except AssertionError:
        #     err_num += 1
print(err_num)
