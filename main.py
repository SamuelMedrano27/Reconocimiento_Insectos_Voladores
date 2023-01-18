import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.models import load_model

print("Librerias Leidas")

# Crear Interfaz

ventana = tk.Tk()
ventana.title("CNN Insectos Voladores")
#ventana.geometry('450x650')
ventana.resizable(0, 0)
ventana.config(bg="#505653")
#Colocar al centro

wtotal = ventana.winfo_screenwidth()
htotal = ventana.winfo_screenheight()
#  Guardamos el largo y alto de la ventana
wventana = 450
hventana = 600

#  Aplicamos la siguiente formula para calcular donde debería posicionarse
pwidth = round(wtotal/2-wventana/2)
pheight = round(htotal/2-hventana/2)

#  Se lo aplicamos a la geometría de la ventana
ventana.geometry(str(wventana)+"x"+str(hventana)+"+"+str(pwidth)+"+"+str(pheight))


# Establecer Fondo
fondo = PhotoImage(file="fondo2.png")
lblfondo = Label(ventana, image=fondo).place(x=0, y=0)


# Crear Comando
def buscar():
    global img_tk
    ventana.filename = filedialog.askopenfilename(title="Encuentra tu imagen",
                                                  filetypes=(('archivos jpeg', '*.jpeg'),('archivos jpg', '*.jpg')))
    # label1=Label(ventana,text=ventana.filename)
    # label1.pack()
    imagencargada = Image.open(ventana.filename)
    new_imgcargada = imagencargada.resize((150, 150))
    rendercar = ImageTk.PhotoImage(new_imgcargada)
    img1_cargada = Label(ventana, image=rendercar)
    img1_cargada.image = rendercar
    img1_cargada.place(x=150, y=120)


    # Import Image from the path with size of (150, 150)

    #img =image.load_img(new_imgcargada, target_size=(150, 150))

    # Convert Image to a numpy array
    img = image.img_to_array(new_imgcargada, dtype=np.uint8)


    # Scaling the Image Array values between 0 and 1
    img = np.array(img) / 255.0

    # Plotting the Loaded Image
    plt.title("Loaded Image")
    plt.axis('off')
    plt.imshow(img.squeeze())
    plt.show()

    loaded_best_model = load_model('model.h5')

    # Get the Predicted Label for the loaded Image
    p = loaded_best_model.predict(img[np.newaxis, ...])

    # Label array
    labels = {0: 'Mariposa', 1: 'Libelula', 2: 'Saltamontes', 3: 'Mariquita', 4: 'Mosquito'}

    print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
    maximo_pro=str(np.max(p[0], axis=-1))
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    messagebox.showinfo(message="La imagen que cargo es  una : "+predicted_class,title="Esta es su predicción")
    messagebox.showinfo(message="La probabilidad es : "+ maximo_pro, title="Probabilidad")
    print("Classified:", predicted_class, "\n\n")


# def imagen ():
# img=Image.open('Imagen1.png')
# new_img= img.resize((300,256))
# render = ImageTk.PhotoImage(new_img)
# img1=Label(ventana,image=render)
# img1.image = render
# img1.place(x=70 , y=70)


def mensaje():
    salir = messagebox.askquestion("Salir", "Desea salir del CNN")
    if salir == 'yes':
        ventana.quit()
        ventana.destroy()


# Crear Botones
boton1 = tk.Button(ventana, command=buscar, text="Cargar Imagen", height=2, width=20)
boton1.place(x=150, y=500)

boton2 = tk.Button(ventana, command=mensaje, text="Salir", height=2, width=20)
boton2.place(x=150, y=550)

# boton3 =tk.Button(ventana,text = "Predicción",height=2 , width=20)
# boton3.place(x=170 , y = 550)


ventana.mainloop()
