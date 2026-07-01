from ultralytics import YOLO
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import arquivos
from formas import *
import easyocr


Img = cv.typing.MatLike
# MODELO YOLO SEGMENTATION
model = YOLO("yolov8n-seg.pt")

reader = easyocr.Reader(['en'])  # 'en' para inglês (números também)


def min_max(a, b):
    if a < b:
        return a, b
    return b, a


def preprocessamentoCV(img: Img) -> Img:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # melhora contraste
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #gray = clahe.apply(gray)

    #gray = cv.equalizeHist(gray) #ruim
    #gray = np.array(((gray / 255) ** 2) * 255, dtype=gray.dtype) #quase
    #gray = np.array(((((gray / 255) - 0.5)/4) ** 0.33333 + 0.5) * 255, dtype=gray.dtype) #sem controle
    x = (gray / 255)
    forca = 2
    gray = np.array(((x ** forca) / (x ** forca + (1 - x) ** forca)) * 255, dtype=gray.dtype)
    
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

def encontrar_centro_12(imagem):
    #Retorna: (x, y) ou None se não encontrar

    # encontrar a caixa delimitadora do '12'
    resultado = reader.readtext(imagem, allowlist='12', text_threshold=0.5)
    
    for (bbox, texto, confianca) in resultado:
        if texto == '12' and confianca > 0.5:
            return ((bbox[0][0] + bbox[2][0])/2, (bbox[0][1] + bbox[2][1])/2)
    
    # 2a tentativa: procurar número romano "XII"
    resultado = reader.readtext(imagem, text_threshold=0.4)  # um pouco mais sensível
    for (bbox, texto, confianca) in resultado:
    
        texto_clean = ''.join(texto.split()).upper()
        if texto_clean == 'XII' and confianca > 0.4:
            return ((bbox[0][0] + bbox[2][0])/2, (bbox[0][1] + bbox[2][1])/2)

    return None

def esta_12_em_cima(edges, circulo):

    mask_coroa = np.zeros_like(edges, dtype=np.uint8)
    cv.circle(mask_coroa, (circulo.cx, circulo.cy), int(circulo.raio * 1.2), 255, -1)
    cv.circle(mask_coroa, (circulo.cx, circulo.cy), int(circulo.raio * 0.7), 0, -1)
    regiao = cv.bitwise_and(edges, edges, mask=mask_coroa)
    h, w = regiao.shape
    metade_sup = regiao[0:circulo.cy, :]
    metade_inf = regiao[circulo.cy:h, :]
    soma_sup = np.sum(metade_sup > 0)
    soma_inf = np.sum(metade_inf > 0)
    return soma_sup > soma_inf

def calcular_somas_bordas(edges, circulo):
    mask_coroa = np.zeros_like(edges, dtype=np.uint8)
    cv.circle(mask_coroa, (circulo.cx, circulo.cy), int(circulo.raio * 0.95), 255, -1)
    cv.circle(mask_coroa, (circulo.cx, circulo.cy), int(circulo.raio * 0.7), 0, -1)
    regiao = cv.bitwise_and(edges, edges, mask=mask_coroa)
    h, w = regiao.shape
    metade_sup = regiao[0:circulo.cy, :]
    metade_inf = regiao[circulo.cy:h, :]
    return np.sum(metade_sup > 0), np.sum(metade_inf > 0)

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
        return []

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
        tol_raio = 1.15 #era 1.05
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
    def __init__(self, circulos: list[Circulo], circulo: Circulo, relogio: Relogio, horas: int, minutos: int, falho=False, img_processada=None):
        self.circulos = circulos
        self.circulo = circulo
        self.relogio = relogio
        self.horas = horas
        self.minutos = minutos
        self.falho = falho
        # imagem já corrigida (perspectiva + rotação) — é o espaço
        # onde os ponteiros foram detectados, e onde o desenho deve ser feito
        self.img_processada = img_processada
    
    def texto_tempo(self) -> str:
        return f"{self.horas:02d}:{self.minutos:02d}"


def visualizar_leitura(output: Img, resize: int, dados: ResultadoLeitura):
    # Usar a imagem processada (espaço onde os ponteiros foram detectados).
    # Se não estiver disponível (resultado falho), cai para a imagem original.
    if dados.img_processada is not None:
        canvas = dados.img_processada.copy()
    else:
        canvas = resizeImagem(output, resize)

    if not dados.falho:
        for c in dados.circulos:
            c.desenhar(canvas, (50, 50, 50), 2)
        dados.circulo.desenhar(canvas, (0, 255, 0), 2)
        dados.circulo.desenhar_centro(canvas, (0, 255, 0), 3)

        # vermelho (BGR 0,0,255) = minuto  |  azul (BGR 255,0,0) = hora
        dados.relogio.ponteiro_m.desenhar(canvas, (0, 0, 255), 4)
        dados.relogio.ponteiro_h.desenhar(canvas, (255, 0, 0), 4)

        titulo = f"Hora predita: {dados.texto_tempo()}"
        texto_hud = dados.texto_tempo()
    else:
        titulo = "Leitura falhou"
        texto_hud = "--:--"

    cv.rectangle(canvas, (10, 10), (220, 70), (255, 255, 255), -1)
    cv.putText(canvas, texto_hud, (20, 55), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(titulo)
    plt.show()


def melhor_circulo(img: Img):
    def calc(c: Circulo):
        cx_imagem = img.shape[1] / 2
        cy_imagem = img.shape[0] / 2
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
            print("Relógio redondo/elíptico detectado. Corrigindo perspectiva com elipse...")
            elipse = cv.fitEllipse(maior_contorno)
            (cx, cy), (d1, d2), angulo = elipse

            # Calcula os semi-eixos
            raio_maior = max(d1, d2) / 2.0
            raio_menor = min(d1, d2) / 2.0

            # cv.fitEllipse retorna 'angulo' como a orientação do eixo
            # correspondente a d1 (largura). O eixo MAIOR só coincide com
            # 'angulo' quando d1 >= d2; caso contrário o eixo maior está a
            # 90° dele. (O critério antigo, baseado no valor de 'angulo',
            # não tinha relação nenhuma com qual eixo é o maior.)
            if d1 < d2:
                angulo += 90

            # evita divisão por zero / elipses degeneradas
            if raio_menor < 1e-3 or raio_maior < 1e-3:
                circulo_pre_definido = None
            else:
                ang_rad = math.radians(angulo)
                cos_t = math.cos(ang_rad)
                sin_t = math.sin(ang_rad)

                # fator de esticamento do eixo menor até igualar o maior
                k = raio_maior / raio_menor

                # transformação AFIM (não projetiva) que estica apenas a
                # direção perpendicular ao eixo maior, mantendo o eixo maior
                # intocado -- monta uma elipse alinhada aos eixos, escala o
                # eixo menor e desfaz a rotação. Evita usar findHomography
                # (que é uma ferramenta para 4 correspondências de pontos
                # ruidosas, não para uma deformação afim conhecida) e evita
                # distorções projetivas indesejadas fora da região central.
                a00 = cos_t**2 + k * sin_t**2
                a01 = sin_t * cos_t * (1 - k)
                a10 = sin_t * cos_t * (1 - k)
                a11 = sin_t**2 + k * cos_t**2

                A = np.array([[a00, a01], [a10, a11]], dtype=np.float64)
                centro = np.array([cx, cy], dtype=np.float64)
                t = centro - A @ centro

                M = np.array([
                    [A[0, 0], A[0, 1], t[0]],
                    [A[1, 0], A[1, 1], t[1]]
                ], dtype=np.float32)

                altura, largura = img.shape[:2]
                img = cv.warpAffine(img, M, (largura, altura), flags=cv.INTER_CUBIC)
                mask_segmentacao = cv.warpAffine(mask_segmentacao, M, (largura, altura), flags=cv.INTER_NEAREST)

                # depois de esticar, a elipse virou um círculo de raio = raio_maior
                circulo_pre_definido = Circulo(int(cx), int(cy), int(raio_maior))

    # remove fundo
    sem_fundo, mask = removerFundo(img, mask_segmentacao)

    gray = preprocessamentoCV(sem_fundo)
    
    if circulo_pre_definido is not None:
        circulo = circulo_pre_definido
        circulos = [circulo] 
    else:
        # para relógios redondos
        circulos = detectarCirculos(gray)
        if len(circulos) == 0:
            print("Círculos não encontrados")
            return ResultadoLeitura(circulos, None, None, None, None, True)
    circulo = sorted(circulos, key=melhor_circulo(img))[0]

    
    # --- DETECÇÃO DO 12
    posicao_12 = encontrar_centro_12(img) #passa a colorida por causa do OCR

    if posicao_12 is not None:
        cx_12, cy_12 = posicao_12

        # 12 tem que estar perto da BORDA do círculo
        dist_12 = math.dist((cx_12, cy_12), (circulo.cx, circulo.cy))
        if not (circulo.raio * 0.65 <= dist_12 <= circulo.raio * 1.15):
            print(f"  -> '12' encontrado fora do raio esperado, ignorando!")
            posicao_12 = None

    if posicao_12 is not None:
        cx_12, cy_12 = posicao_12
        
        #g = gray.copy()
        #circ = Circulo(int(cx_12), int(cy_12), 1)
        #circ.desenhar_centro(g, (255, 0, 255), 5)
        #plt.imshow(g)
        #plt.show()

        # vetor do centro do relógio para o 12
        vetor_x = cx_12 - circulo.cx
        vetor_y = cy_12 - circulo.cy
        
        # ângulo do 12 em graus
        angulo_atual = math.degrees(math.atan2(vetor_y, vetor_x))
        
        # para cima
        angulo_alvo = -90
        
        # quanto precisamos rotacionar
        #angulo_rotacao = angulo_atual - angulo_alvo
        angulo_rotacao = (angulo_atual - angulo_alvo) % 360
        if angulo_rotacao > 180:
            angulo_rotacao -= 360
        
        print(f"  -> Ângulo do 12: {angulo_atual:.2f}°. Rotacionando: {angulo_rotacao:.2f}° para alinhar")
        
        # rotaciona
        if abs(angulo_rotacao) > 3.0:  # só se o desvio for significativo
            print(f"DEBUG: {circulo=}")
            print(f"DEBUG: {circulo.cx=}, {circulo.cy=}")

            if not isinstance(angulo_rotacao, (int, float)):
                print(f"  -> Aviso: angulo_rotacao não é numérico ({angulo_rotacao}), ignorando rotação.")
                angulo_rotacao = 0.0

            M = cv.getRotationMatrix2D((float(circulo.cx), float(circulo.cy)), angulo_rotacao, 1.0)
            img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_CUBIC)
            mask_segmentacao = cv.warpAffine(mask_segmentacao, M, (mask_segmentacao.shape[1], mask_segmentacao.shape[0]), flags=cv.INTER_NEAREST)
            
            # recalcula o gray
            sem_fundo, _ = removerFundo(img, mask_segmentacao)
            gray = preprocessamentoCV(sem_fundo)
    else:
        print("  -> 12 não detectado. Verificando se está de cabeça para baixo")
        #compara as bordas
        edges_temp = cv.Canny(gray, 50, 150)
        if esta_12_em_cima(edges_temp, circulo) is False:
            # Só gira se tiver certeza que está invertido
            soma_sup, soma_inf = calcular_somas_bordas(edges_temp, circulo)  # função auxiliar
            if soma_inf > soma_sup * 1.5:  # só gira se a diferença for grande
                print("    Evidência forte: relógio de cabeça para baixo. Rotacionando 180°.")
                img = cv.rotate(img, cv.ROTATE_180)
                mask_segmentacao = cv.rotate(mask_segmentacao, cv.ROTATE_180)
                circulo.cx = img.shape[1] - circulo.cx
                circulo.cy = img.shape[0] - circulo.cy
                sem_fundo, _ = removerFundo(img, mask_segmentacao)
                gray = preprocessamentoCV(sem_fundo)
            else:
                print("    Evidência fraca. Mantendo orientação atual.")
        else:
            print("    Relógio parece já estar em pé. Mantendo orientação.")

    # salva a imagem já totalmente corrigida — é nesse espaço que os ponteiros
    # serão detectados, e é nesse espaço que o desenho deve acontecer
    img_processada = img.copy()

    # máscara circular interna
    mask_clock = np.zeros_like(gray)
    cv.circle(mask_clock, (circulo.cx, circulo.cy), int(circulo.raio * 0.92), 255, -1)
    gray = cv.bitwise_and(gray, gray, mask=mask_clock)
    
    # bordas
    otsu_thresh, _ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    low = max(0, int(otsu_thresh * 0.5))
    high = min(255, int(otsu_thresh * 1.5))
    edges = cv.Canny(gray, low, high)

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

    # se faltam ponteiros, tenta de novo com bordas mais sensíveis
    if len(clusters) < 2:
        print("Poucos ponteiros no 1º passe. Tentando detecção mais sensível...")
        edges2 = cv.Canny(gray, max(0, low // 2), high)
        edges2 = cv.dilate(edges2, kernel, iterations=1)
        edges2 = cv.morphologyEx(edges2, cv.MORPH_CLOSE, kernel)
        linhas2 = cv.HoughLinesP(edges2, rho=1, theta=np.pi / 180,
                                 threshold=25, minLineLength=int(circulo.raio * 0.3),
                                 maxLineGap=20)
        if linhas2 is not None:
            linhas2 = [Linha(*l[0]) for l in linhas2]
            ponteiros2 = filtrarLinhas(linhas2, circulo)
            clusters2 = clusterizarPonteiros(ponteiros2, circulo.raio)
            if len(clusters2) >= 2:
                clusters = clusters2

    if len(clusters) < 2:
        print("Ponteiros insuficientes")
        return ResultadoLeitura(circulos, circulo, None, None, None, True, img_processada)

    relogio = Relogio(circulo, *clusters)
    horas, minutos = relogio.calcular_hora()

    return ResultadoLeitura(circulos, circulo, relogio, horas, minutos, img_processada=img_processada)