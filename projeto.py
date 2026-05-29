from ultralytics import YOLO
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import arquivos
import visao


ARTE_CAPA = '''
·▄▄▄▄  ▄▄▄ .▄▄▄▄▄▄▄▄ . ▄▄· ▄▄▄▄▄      ▄▄▄    ·▄▄▄▄  ▄▄▄ .   ▄ .▄      ▄▄▄   ▄▄▄· .▄▄ · 
██· ██ ▀▄.▀·•██  ▀▄.▀·▐█ ▌▪•██   ▄█▀▄ ▀▄ █·  ██· ██ ▀▄.▀·  ██▪▐█ ▄█▀▄ ▀▄ █·▐█ ▀█ ▐█ ▀. 
▐█▪ ▐█▌▐▀▀▪▄ ▐█.▪▐▀▀▪▄██ ▄▄ ▐█.▪▐█▌.▐▌▐▀▀▄   ▐█▪ ▐█▌▐▀▀▪▄  ██▀▀█▐█▌.▐▌▐▀▀▄ ▄█▀▀█ ▄▀▀▀█▄
██. ██ ▐█▄▄▌ ▐█▌·▐█▄▄▌▐███▌ ▐█▌·▐█▌.▐▌▐█•█▌  ██. ██ ▐█▄▄▌  ██▌▐▀▐█▌.▐▌▐█•█▌▐█▪ ▐▌▐█▄▪▐█
▀▀▀▀▀•  ▀▀▀  ▀▀▀  ▀▀▀ ·▀▀▀  ▀▀▀  ▀█▄▀▪.▀  ▀  ▀▀▀▀▀•  ▀▀▀   ▀▀▀ · ▀█▄▀▪.▀  ▀ ▀  ▀  ▀▀▀▀ 
  ╔═══════╗
 ╔╝   │   ╚╗
╔╝    │    ╚╗
║     └───  ║
╚╗         ╔╝
 ╚╗       ╔╝
  ╚═══════╝'''


def deteccao_geral():
    pasta_alvo = "imagens"

    lista = arquivos.listar_imagens(pasta_alvo)
    
    if not lista:
        print("Nenhuma imagem encontrada.")
        exit()

    print(f"{len(lista)} imagens encontradas.")

    erros = 0
    for caminho in lista:
        print(f"\nProcessando: ...{caminho[50:]}")
        img = cv.imread(caminho)

        if img is None:
            print("Erro ao carregar imagem.")
            continue

        resultado_deteccao = visao.detectarRelogio(img)
        
        if resultado_deteccao.falho:
            erros += 1
            print("Nenhum relógio detectado.")
            continue
        
        visao.visualizar_deteccao(img.copy(), resultado_deteccao)

        resize_leitura = 500
        resultado_hora = visao.lerRelogio(resultado_deteccao.crop, resize_leitura, resultado_deteccao.mask)

        if resultado_hora.falho:
            erros += 1
            print("Falha na leitura do relógio.")
            continue
        
        output = resultado_deteccao.crop.copy()
        visao.visualizar_leitura(output, resize_leitura, resultado_hora)
        print(f"Resultado final: {resultado_hora.texto_tempo()}")
        arquivos.salvar_imagem(output, caminho)

    
    n_acertos = len(lista) - erros
    print(f'({(n_acertos) * 100/len(lista):.2f}%) {n_acertos}/{len(lista)} das imagens funcionaram.')


if __name__ == "__main__":
    print(ARTE_CAPA)
    print('Seja bem vindo ao nosso trabalho de tópicos I')
    print('- Bruna, Gabriela, Felipe')
    print()

    sair = False
    while not sair:
        print('Digite "g" para ver a detecção de todos os relógios um por um.')
        print('Digite "f" para calcular o resultado em todos os relógios')
        print('Digite qualquer outra coisa para sair.')

        comando = input()
        if comando == 'g':
            deteccao_geral()
        elif comando == 'f':
            pass
        else:
            sair = True

    

