from ultralytics import YOLO
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import arquivos
from formas import *


Img = cv.typing.MatLike
# MODELO YOLO SEGMENTATION
model = YOLO("yolov8n-seg.pt")


def min_max(a, b):
    if a < b:
        return a, b
    return b, a


def preprocessamentoCV(img: Img) -> Img:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # melhora contraste
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    gray = clahe.apply(gray)

    # preserva bordas
    gray = cv.bilateralFilter(gray, 7, 50, 50)

    return gray


def resizeImagem(img: Img, largura=500) -> Img:
    h, w = img.shape[:2]
    escala = largura / w    
    nova_altura = int(h * escala)

    return cv.resize(img, (largura, nova_altura))


class ResultadoDeteccao:
    def __init__(self, crop: Img = None, mask: Img = None, bbox: tuple = None):
        self.crop = crop
        self.mask = mask
        self.bbox = bbox
        self.falho = mask is None or crop is None


def visualizar_deteccao(output: Img, dados: ResultadoDeteccao):
    x1, y1, x2, y2 = dados.bbox
    cv.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 3)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Relógio detectado")
    plt.show()



def detectarRelogio(img: Img) -> ResultadoDeteccao:
    results = model(img, verbose=False)[0]

    if results.masks is None:
        return ResultadoDeteccao()

    melhor_score = 0
    melhor_crop = None
    melhor_mask = None
    melhor_box = None

    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        nome = model.names[cls]

        if nome != "clock":
            continue

        score = float(box.conf[0])

        if score < 0.3:
            continue

        if score > melhor_score:
            melhor_score = score

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # proteção
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            crop = img[y1:y2, x1:x2]

            # máscara segmentada
            mask = results.masks.data[i].cpu().numpy()
            mask = cv.resize(mask, (img.shape[1], img.shape[0]))

            mask_crop = mask[y1:y2, x1:x2]
            melhor_crop = crop
            melhor_mask = mask_crop
            melhor_box = (x1, y1, x2, y2)

    return ResultadoDeteccao(melhor_crop, melhor_mask, melhor_box)


def removerFundo(img: Img, mask):
    mask_bin = (mask > 0.5).astype(np.uint8) * 255

    # suavizar máscara
    mask_bin = cv.GaussianBlur(mask_bin, (7,7), 0)

    resultado = cv.bitwise_and(img, img, mask=mask_bin)

    return resultado, mask_bin


def detectarCirculos(gray: Img) -> list[Circulo]:
    h, w = gray.shape[:2]

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=h // 2,
        param1=120,
        param2=30,
        minRadius=int(min(h, w) * 0.30),
        maxRadius=int(min(h, w) * 0.48)
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # pega maior círculo
    #return Circulo(*max(circles, key=lambda c: c[2]))
    return [Circulo(*circle) for circle in circles]


def detectarLinhas(edges) -> list[Linha]:
    linhas = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=45, minLineLength=40, maxLineGap=10)
    if len(linhas) == 0:
        raise(Exception('sem linhas detectadas')) # -------------------------------------------- teste
    return [Linha(*l[0]) for l in linhas]


def filtrarLinhas(linhas:list[Linha], circulo:Circulo) -> list[Linha]:
    if linhas is None:
        return []

    ponteiros = []

    for linha in linhas:
        # remove linhas pequenas
        if linha.get_comprimento() < circulo.raio * 0.35:
            continue
        
        d1 = math.dist((linha.x1, linha.y1), (circulo.cx, circulo.cy))
        d2 = math.dist((linha.x2, linha.y2), (circulo.cx, circulo.cy))

        d_centro = min(d1, d2)
        d_ponta = max(d1, d2)

        # uma ponta precisa tocar centro
        tol_centro = 0.25
        if d_centro > circulo.raio * tol_centro:
            continue
        
        # ponta da linha deve ficar dentro do relógio
        tol_raio = 1.05
        if d_ponta > circulo.raio * tol_raio:
            continue
        
        ponteiros.append(linha)

    return ponteiros


def clusterizarPonteiros(ponteiros: list[Linha], raio: float, tolerancia=8) -> tuple[Linha]:
    if not ponteiros:
        return []

    ponteiros.sort(key=lambda p: p.get_angulo())
    
    clusters = [[ponteiros[0]]]

    for i in range(1, len(ponteiros)):
        atual = ponteiros[i]
        anterior = ponteiros[i - 1]

        if abs(atual.get_angulo() - anterior.get_angulo()) <= tolerancia:
            clusters[-1].append(atual)
        else:
            clusters.append([atual])


    candidatos = []

    for cluster in clusters:
        angulos = [c.get_angulo() for c in cluster]
        tamanhos = [c.get_comprimento() for c in cluster]

        # maior linha
        coords = max(cluster, key=lambda c: c.get_comprimento())
        tamanho_medio = np.mean(tamanhos)
        razao = tamanho_medio / raio

        # remove ponteiros MUITO longos
        # normalmente segundos
        if razao > 0.92:
            continue

        # remove linhas pequenas
        if razao < 0.35:
            continue
        
        candidatos.append(linha_media(cluster))


    if len(candidatos) < 2:
        return []

    # ESCOLHA DOS PONTEIROS
    candidatos = sorted(candidatos, key=lambda c: -c.get_comprimento())
    
    # minuto = maior restante
    ponteiro_minuto = candidatos[0]

    # hora = menor que minuto
    ponteiro_hora = None

    for c in candidatos[1:]:
        # ponteiro de hora deve ser significativamente menor
        if c.get_comprimento() < ponteiro_minuto.get_comprimento() * 0.8:
            ponteiro_hora = c
            break
    
    # fallback
    if ponteiro_hora is None:
        ponteiro_hora = candidatos[1]

    return ponteiro_minuto, ponteiro_hora


class ResultadoLeitura:
    def __init__(self, circulos: list[Circulo], circulo: Circulo, relogio: Relogio, horas: int, minutos: int, falho=False):
        self.circulos = circulos
        self.circulo = circulo
        self.relogio = relogio
        self.horas = horas
        self.minutos = minutos
        self.falho = falho
    
    def texto_tempo(self) -> str:
        return f"{self.horas:02d}:{self.minutos:02d}"


def visualizar_leitura(output:Img, resize: int, dados: ResultadoLeitura):
    output = resizeImagem(output, resize)
    for c in dados.circulos:
        c.desenhar(output, (50, 50, 50), 2)
    dados.circulo.desenhar(output, (0, 255, 0), 2)
    dados.circulo.desenhar_centro(output, (0, 255, 0), 3)
    
    dados.relogio.ponteiro_m.desenhar(output, (0, 0, 255), 4)
    dados.relogio.ponteiro_h.desenhar(output, (255, 0, 0), 4)

    cv.rectangle(output, (10, 10), (220, 70), (255,255,255), -1)
    cv.putText(output, dados.texto_tempo(), (20, 55), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

    plt.figure(figsize=(8,8))
    plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Hora predita: {dados.texto_tempo()}")
    plt.show()


def melhor_circulo(img: Img):
    def calc(c: Circulo):
        ponto_central = img.shape[0] / 2, img.shape[1] / 2
        return math.dist((c.cx, c.cy), ponto_central) * 2 - c.raio
    return calc


def lerRelogio(img: Img, resize: int, mask_segmentacao: Img) -> ResultadoLeitura:
    img = resizeImagem(img, resize)
    mask_segmentacao = cv.resize(mask_segmentacao, (img.shape[1], img.shape[0]))

    # remove fundo
    sem_fundo, mask = removerFundo(img, mask_segmentacao)

    gray = preprocessamentoCV(sem_fundo)
    circulos = detectarCirculos(gray)
    if len(circulos) == 0:
        print("Círculos não encontrados")
        return ResultadoLeitura(circulos, None, None, None, None, True)

    circulo = sorted(circulos, key=melhor_circulo(img))[0]

    # máscara circular interna
    mask_clock = np.zeros_like(gray)
    cv.circle(mask_clock, (circulo.cx, circulo.cy), int(circulo.raio * 0.92), 255, -1)
    gray = cv.bitwise_and(gray, gray, mask=mask_clock)
    
    # bordas
    edges = cv.Canny(gray, 70, 180)
    kernel = np.ones((3,3), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    linhas = detectarLinhas(edges)
    #---------------------------------
    '''k = 0
    img_li = img.copy()
    for l in linhas:
        l.desenhar(img_li, (255, 0, k), 2)
        k += 31
        if k > 255:
            k -= 255
    plt.imshow(cv.cvtColor(img_li, cv.COLOR_BGR2RGB))
    plt.show()'''
    #---------------------------------

    ponteiros = filtrarLinhas(linhas, circulo)
    clusters = clusterizarPonteiros(ponteiros, circulo.raio)

    if len(clusters) < 2:
        print("Ponteiros insuficientes")
        return ResultadoLeitura(circulos, circulo, None, None, None, True)

    relogio = Relogio(circulo, *clusters)
    horas, minutos = relogio.calcular_hora()

    dados = ResultadoLeitura(circulos, circulo, relogio, horas, minutos)

    return dados