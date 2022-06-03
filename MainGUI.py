import tkinter
from tkinter import *
from tkinter import filedialog
import SignatureInit

def UploadAction(event=None):
    filepath = filedialog.askopenfilename()
    print('Selected:', filepath)
    SignatureInit.initTesting(filepath)
    
        
  
    
root=tkinter.Tk()
root.title("SIGNATURE VERIFICATION SYSTEM")
root.geometry("500x500")



button1=tkinter.Button(root, text="INPUT SIGNATURE IMAGE",height=1,width=25,command=UploadAction)
button1.place(x=200, y=200)



root.mainloop()

