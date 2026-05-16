from ultralytics import YOLO
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import arquivos
import visao



if __name__ == "__main__":
    pasta_alvo = "imagens"

    lista = arquivos.listar_imagens(pasta_alvo)
    
    # VERIFICA SE EXISTEM IMAGENS
    if not lista:
        print("Nenhuma imagem encontrada.")
        exit()

    print(f"{len(lista)} imagens encontradas.")

    # LOOP NAS IMAGENS
    for caminho in lista:
        print(f"\nProcessando: {caminho}")
        img = cv.imread(caminho)

        if img is None:
            print("Erro ao carregar imagem.")
            continue

        # DETECÇÃO DO RELÓGIO
        resultado = visao.detectarRelogio(img)

        if resultado is None:
            print("Nenhum relógio detectado.")
            continue

        crop, mask, bbox = resultado

        x1, y1, x2, y2 = bbox

        img_box = img.copy()

        cv.rectangle(img_box, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # VISUALIZAR DETECÇÃO
        rgb = cv.cvtColor(img_box, cv.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("Relógio detectado")
        plt.show()

        resultado_hora = visao.lerRelogio(crop, mask, caminho)

        if resultado_hora is not None:
            horas, minutos = resultado_hora
            print(f"Resultado final: {horas:2d}:{minutos:2d}")

        else:
            print("Falha na leitura do relógio.")


