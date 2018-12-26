import tkinter
def display_dialog():
    dialog = tkinter.Toplevel()
    big_frame = tkinter.Frame(dialog)
    big_frame.pack(fill='both', expand=True)

    label = tkinter.Label(big_frame, text="Hello World")
    label.place(relx=0.5, rely=0.3, anchor='center')
    dialog.transient(root)
    dialog.geometry('300x150')
    dialog.wait_window()


root = tkinter.Tk()
big_frame = tkinter.Frame(root)
big_frame.pack(fill='both', expand=True)

button = tkinter.Button(big_frame, text="Click me", command=display_dialog)
button.pack()
root.mainloop()