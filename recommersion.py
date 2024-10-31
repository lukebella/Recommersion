"""import tkinter

class Main(tkinter.Tk):
    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)


if __name__=='__main__':
    Main().mainloop()"""


from tkinter import *

window = Tk()
window.geometry("900x550")
window.title("RECOMMERSION")
window.grid_columnconfigure(0, weight=1)
welcome_label = Label(window,
                         text="Welcome! Aggiungi una parola o una frase da scaricare:",
                         font=("Helvetica", 15))
welcome_label.grid(row=0, column=0, sticky="N", padx=20, pady=10)

def download_ascii():
    text_response = "Aggiungi una parola o una frase al campo input!"

download_button = Button(text="DOWNLOAD ASCII ART", command=download_ascii)


frame = Frame(window, width = 50, height = 50, bg = "#ffffff")
frame.grid(row = 0, column = 1, padx = 1, pady = 1)
# https://www.youtube.com/watch?v=DGeDcxul5Zk



if __name__ == "__main__":
    window.mainloop()