import os
import tkinter as tk

win = tk.Tk()
win.title('网易云音乐缓存解码器')
win.geometry('300x400+500+300')
win.resizable(width=False, height=True)

def count_row():
    num = 0
    while True:
        num += 23
        yield num
row_num = count_row()

tk.Label(win, text='步骤:', font=('黑体', 12)).place(x=0, y=0)
tk.Label(win, text='1、打开网易云音乐的缓存文件夹').place(x=0, y=next(row_num))
tk.Label(win, text='2、将.uc文件复制到程序同一文件夹下').place(x=0, y=next(row_num))
tk.Label(win, text='3、点击确定按钮').place(x=0, y=next(row_num))

def overload(fun):
    def callback():
        fname_num = fun()
        if fname_num == 0:
            tk.Label(win, text='工作目录下未含有待破解文件').place(x=66, y=next(row_num))
        else:
            tk.Label(win, text='解码{}首歌'.format(fname_num)).place(x=120, y=next(row_num))
    return callback

@overload
def main():
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
    return fname_num

tk.Button(win, text='确定', command=main).place(x=130, y=next(row_num)-6)
tk.mainloop()