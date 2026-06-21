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
    #antes estava: threshold=45, minLineLength=40, maxLineGap=10
    linhas = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=45, minLineLength=40, maxLineGap=10)
    if linhas is None or len(linhas) == 0:
        # mais sensível
        linhas = cv.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=30, minLineLength=25, maxLineGap=15)
    if linhas is None or len(linhas) == 0:
        #raise(Exception('sem linhas detectadas')) # -------------------------------------------- teste
        return []  # não quebra
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

    ponteiros = sorted(ponteiros, key=lambda p: p.get_angulo())
    
    clusters = [[ponteiros[0]]]

    for i in range(1, len(ponteiros)):
        atual = ponteiros[i]
        anterior = ponteiros[i - 1]

        if abs(atual.get_angulo() - anterior.get_angulo()) <= tolerancia:
            clusters[-1].append(atual)
        else:
            clusters.append([atual])
    if len(clusters) > 1:
        ang_primeiro = clusters[0][0].get_angulo()
        ang_ultimo = clusters[-1][-1].get_angulo()
        distancia_circular = (180 - ang_ultimo) + ang_primeiro
        if distancia_circular <= tolerancia:
            clusters[0] = clusters.pop() + clusters[0]

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
    
    # garantia
    if ponteiro_hora is None:
        ponteiro_hora = candidatos[1]

    return ponteiro_minuto, ponteiro_hora

def estimar_espessura_ponteiro(mask_relogio_bin: Img, ponteiro: Linha, circulo: Circulo, amostras=5) -> float:
    # mede quantos pixels da máscara tem na direção PERPENDICULAR ao ponteiro (espessura em pixels)
    dx = ponteiro.x2 - ponteiro.x1
    dy = ponteiro.y2 - ponteiro.y1
    comprimento = math.hypot(dx, dy)
    if comprimento == 0:
        return 0.0
 
    # vetor perpendicular normalizado
    px, py = -dy / comprimento, dx / comprimento
 
    espessuras = []
    for t in np.linspace(0.2, 0.8, amostras):
        cx_amostra = ponteiro.x1 + dx * t
        cy_amostra = ponteiro.y1 + dy * t
 
        # caminha pra cada lado perpendicular até achar a borda da máscara
        passo = 0
        max_passo = int(circulo.raio * 0.15)  # não deveria precisar de mais que isso
        while passo < max_passo:
            xa = int(cx_amostra + px * passo)
            ya = int(cy_amostra + py * passo)
            xb = int(cx_amostra - px * passo)
            yb = int(cy_amostra - py * passo)
 
            dentro_a = (0 <= ya < mask_relogio_bin.shape[0] and 0 <= xa < mask_relogio_bin.shape[1]
                        and mask_relogio_bin[ya, xa] > 0)
            dentro_b = (0 <= yb < mask_relogio_bin.shape[0] and 0 <= xb < mask_relogio_bin.shape[1]
                        and mask_relogio_bin[yb, xb] > 0)
 
            if not dentro_a and not dentro_b:
                break
            passo += 1
 
        espessuras.append(passo * 2)
 
    return float(np.mean(espessuras)) if espessuras else 0.0

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
        cx_imagem = img.shape[1] / 2
        cy_imagem = img.shape[0] / 2
        ponto_central = img.shape[0] / 2, img.shape[1] / 2
        return math.dist((c.cx, c.cy), (cx_imagem, cy_imagem)) * 2 - c.raio
    return calc

def ordenar_pontos(pontos):
    #Ordena: [superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo]
    ret = np.zeros((4, 2), dtype="float32")
    
    soma = pontos.sum(axis=1)
    ret[0] = pontos[np.argmin(soma)]  # Superior-Esquerdo
    ret[2] = pontos[np.argmax(soma)]  # Inferior-Direito
    
    diff = np.diff(pontos, axis=1)
    ret[1] = pontos[np.argmin(diff)]  # Superior-Direito
    ret[3] = pontos[np.argmax(diff)]  # Inferior-Esquerdo
    
    return ret

def lerRelogio(img: Img, resize: int, mask_segmentacao: Img) -> ResultadoLeitura:
    img = resizeImagem(img, resize)
    mask_segmentacao = cv.resize(mask_segmentacao, (img.shape[1], img.shape[0]))
    circulo_pre_definido = None

    if mask_segmentacao is not None:
        # converte para 8 bits se necessário (para usar no findContours)
        if mask_segmentacao.dtype != np.uint8: #precisamos de 8 bits
            mask_para_contorno = (mask_segmentacao > 0.5).astype(np.uint8) * 255
        else:
            mask_para_contorno = mask_segmentacao

        contornos, _ = cv.findContours(mask_para_contorno, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contornos:
            maior_contorno = max(contornos, key=cv.contourArea)
            
            area = cv.contourArea(maior_contorno)
            
            # Opção 1: Tentar encontrar um polígono (para quadrados/retângulos)
            epsilon = 0.02 * cv.arcLength(maior_contorno, True)
            approx = cv.approxPolyDP(maior_contorno, epsilon, True)

            if len(approx) == 4:
                print("Relógio quadrado/retângulo detectado. Corrigindo perspectiva...")
                
                pts_origem = ordenar_pontos(approx.reshape(4, 2))
                
                (tl, tr, br, bl) = pts_origem
                largura = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
                altura = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
                
                pts_destino = np.float32([
                    [0, 0],
                    [largura - 1, 0],
                    [largura - 1, altura - 1],
                    [0, altura - 1]
                ])
                
                # homografia
                h, _ = cv.findHomography(pts_origem, pts_destino, cv.RANSAC)
                if h is not None:
                    img = cv.warpPerspective(img, h, (largura, altura))
                    mask_segmentacao = cv.warpPerspective(mask_segmentacao, h, (largura, altura))
                    
                    # cria um círculo artificial
                    cx = largura // 2
                    cy = altura // 2
                    # diametro = 90% do menor lado (evita bordas)
                    raio = int(min(largura, altura) * 0.45)
                    circulo_pre_definido = Circulo(cx, cy, raio)
            
            # Opção 2: Se relógio não for quadrado, usa a elipse (para redondos)
            elif len(maior_contorno) >= 5 and area > 1000:
                print("Relógio redondo detectado. Corrigindo perspectiva com elipse...")
                ellipse = cv.fitEllipse(maior_contorno)
                (cx, cy), (d1, d2), angulo = ellipse

                # Calcula os semi-eixos
                raio_maior = max(d1, d2) / 2.0
                raio_menor = min(d1, d2) / 2.0
                
                if angulo > 90:
                    angulo = angulo - 90
                else:
                    angulo = angulo + 90
                
                ang_rad = math.radians(angulo)
                
                # Pontos dos eixos maior e menor da elipse
                p1x = int(cx + raio_maior * math.cos(ang_rad))
                p1y = int(cy + raio_maior * math.sin(ang_rad))
                p2x = int(cx - raio_maior * math.cos(ang_rad))
                p2y = int(cy - raio_maior * math.sin(ang_rad))
                p3x = int(cx + raio_menor * math.cos(ang_rad + math.pi/2))
                p3y = int(cy + raio_menor * math.sin(ang_rad + math.pi/2))
                p4x = int(cx - raio_menor * math.cos(ang_rad + math.pi/2))
                p4y = int(cy - raio_menor * math.sin(ang_rad + math.pi/2))
                
                pts_origem = np.float32([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]])
                
                # Pontos de elipse para circulo
                pts_destino = np.float32([
                    [int(cx + raio_maior * math.cos(ang_rad)), int(cy + raio_maior * math.sin(ang_rad))],
                    [int(cx - raio_maior * math.cos(ang_rad)), int(cy - raio_maior * math.sin(ang_rad))],
                    [int(cx + raio_maior * math.cos(ang_rad + math.pi/2)), int(cy + raio_maior * math.sin(ang_rad + math.pi/2))],
                    [int(cx - raio_maior * math.cos(ang_rad + math.pi/2)), int(cy - raio_maior * math.sin(ang_rad + math.pi/2))]
                ])
                
                h, _ = cv.findHomography(pts_origem, pts_destino, cv.RANSAC)
                if h is not None:
                    altura, largura = img.shape[:2]
                    img_corrigida = cv.warpPerspective(img, h, (largura, altura))
                    mask_corrigida = cv.warpPerspective(mask_segmentacao, h, (largura, altura))
                    
                    img = img_corrigida
                    mask_segmentacao = mask_corrigida                

    # remove fundo
    sem_fundo, mask = removerFundo(img, mask_segmentacao)

    gray = preprocessamentoCV(sem_fundo)
    if circulo_pre_definido is not None:
        circulo = circulo_pre_definido
        circulos = [circulo] 
    else:
        # para relógios redondos
        circulos = detectarCirculos(gray)
        if circulos is None or len(circulos) == 0:
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