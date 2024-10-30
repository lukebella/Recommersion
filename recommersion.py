"""import tkinter

class Main(tkinter.Tk):
    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)


if __name__=='__main__':
    Main().mainloop()"""


import tkinter as tk
import requests

window = tk.Tk()
window.geometry("900x550")
window.title("ASCII ART DOWNLOADER")
window.grid_columnconfigure(0, weight=1)
welcome_label = tk.Label(window,
                         text="Welcome! Aggiungi una parola o una frase da scaricare:",
                         font=("Helvetica", 15))
welcome_label.grid(row=0, column=0, sticky="N", padx=20, pady=10)

def download_ascii():
    text_response = "Aggiungi una parola o una frase al campo input!"

download_button = tk.Button(text="DOWNLOAD ASCII ART", command=download_ascii)
download_button.bbox(row=2, column=0, sticky="WE", pady=30, padx=30)



if __name__ == "__main__":
    window.mainloop()