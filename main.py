import cv2

# Plantillas de OpenCV (rostros, ojos, sonrisa)
face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_rigth = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
eye_left = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')

# Enciende la camara '0' (depende de las camaras instaladas)
cap = cv2.VideoCapture(0)

while (True):
    # Se lee la captura de la camara, devuelve dos objetos
    # valido contiene un booleano
    # img contiene el frame de la captura
    valido, img = cap.read()

    # Si el frame es true
    if valido:
        # Combirtiendo imagen en escala de grises
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Con el objeto creado con CascadeClassifier con el modelo que detecta rostros
        # se analiza la imagen del frame especificando en cuanto escalar la imagen
        # y la cantidad de vecinos cercanos para retenerlo
        rostros = face.detectMultiScale(img_gris, 1.3, 5)

        # Se recorre el objeto con los datos de deteccion separandolos
        # y con el metodo rectangle se dibuja un rectangulo sobre el frame
        for (x, y, w, h) in rostros:
            cv2.rectangle(img, (x, y), (x + w, y + h), (125, 255, 0), 2)
            # detectando ojo derecho y dibujando el rectangulo
            eyer = eye_rigth.detectMultiScale(img_gris, 1.3, 5)
            for (rx, ry, rw, rh) in eyer:
                cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (40, 55, 200), 2)
            # detectando ojo izquierdo y dibujando el rectangulo
            eyel = eye_left.detectMultiScale(img_gris, 1.3, 5)
            for (lx, ly, lw, lh) in eyel:
                cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (40, 55, 200), 2)
            # detectando sonrisa y dibujando el rectangulo
            s = smile.detectMultiScale(img_gris, 1.7, 20)
            for (sx, sy, sw, sh) in s:
                cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (40, 55, 200), 2)
        # Mostrar los frames en una ventana
        cv2.imshow('img', img)

        # Con la tecla 'q' salimos del programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Termina la captura y cierra las ventanas que se hayan creado
cap.release()
cv2.destroyAllWindows()
