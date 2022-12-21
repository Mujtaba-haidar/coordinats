from tkinter import *
from tkinter.ttk import *
from time import strftime

root = Tk()
root.title("Clock")

def time():
     string = strftime("%I:%M:%S %p")
     label.config(text=string)
     label.after(1000,time)
     
label = Label(root,font=("Baskerville Old Face",50), background = "#009bde", foreground="black")
label.pack(anchor="center")
time()

mainloop()