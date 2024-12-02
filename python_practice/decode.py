import os

files = os.listdir(os.getcwd())
fname_num = 0
for fname in files:
    if '.uc' in fname:
        fname_num += 1
        with open(fname, 'rb') as f:
            data = f.read()
        data_arr = bytearray(data)
        for i in range(len(data_arr)):
            data_arr[i] ^= 163
        with open('music'+str(fname_num)+'.mp3', 'wb') as f:
            f.write(bytes(data_arr))

print('解码{}首歌'.format(fname_num))
input()