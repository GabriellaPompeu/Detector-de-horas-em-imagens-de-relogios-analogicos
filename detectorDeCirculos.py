import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import math

# MODELO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")

# PREPROCESSAMENTO CNN
def preprocessamentoCNN(img):
    imagem = img.copy().astype(np.float32)
    imagem = imagem.transpose(2, 0, 1)
    imagem /= 255.0

    return imagem

# PREPROCESSAMENTO OPENCV
def preprocessamentoCV(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    return gray

# RESIZE
def resizeImagem(img, largura=500):
    h, w = img.shape[:2]
    escala = largura / w
    nova_altura = int(h * escala)
    img = cv.resize(img, (largura, nova_altura))

    return img

# DETECTAR RELÓGIO
def detectarRelogio(img, threshold=0.7):
    inp = [torch.from_numpy(preprocessamentoCNN(img)).float().to(device)]

    with torch.no_grad():
        predict = model(inp)[0]

    boxes = predict['boxes'].detach().cpu().numpy()
    labels = predict['labels'].detach().cpu().numpy()
    scores = predict['scores'].detach().cpu().numpy()

    melhor_box = None
    melhor_score = 0

    for i in range(len(labels)):
        # classe 85 = clock no COCO
        if labels[i] == 85 and scores[i] > threshold:
            if scores[i] > melhor_score:
                melhor_score = scores[i]
                melhor_box = boxes[i]

    if melhor_box is None:
        return None

    return melhor_box.round().astype(np.int32)

# DETECTAR CÍRCULO
def detectarCirculo(gray):
    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=int(gray.shape[0] * 0.25),
        maxRadius=int(gray.shape[0] * 0.48)
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # pega maior círculo
    x, y, r = max(circles, key=lambda c: c[2])

    return x, y, r

# DISTÂNCIA
def distancia(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# DETECTAR LINHAS
def detectarLinhas(edges):
    linhas = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )

    return linhas

# FILTRAR LINHAS
def filtrarLinhas(linhas, cx, cy, raio):
    if linhas is None:
        return []

    ponteiros = []

    for linha in linhas:
        x1, y1, x2, y2 = linha[0]

        d1 = distancia(x1, y1, cx, cy)
        d2 = distancia(x2, y2, cx, cy)

        # uma ponta deve estar próxima do centro
        if d1 > raio * 0.25 and d2 > raio * 0.25:
            continue

        # organiza ponto mais próximo do centro
        if d1 > d2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        tamanho = distancia(x1, y1, x2, y2)

        # ignora linhas pequenas
        if tamanho < raio * 0.3:
            continue

        dx = x2 - x1
        dy = y1 - y2

        angulo = math.degrees(math.atan2(dx, dy))

        if angulo < 0:
            angulo += 360

        ponteiros.append({
            'angulo': angulo,
            'tamanho': tamanho,
            'coords': (x1, y1, x2, y2)
        })

    return ponteiros

# CLUSTERIZAÇÃO ANGULAR
def clusterizarPonteiros(ponteiros, tolerancia=8):
    if len(ponteiros) == 0:
        return []

    ponteiros = sorted(ponteiros, key=lambda p: p['angulo'])
    clusters = [[ponteiros[0]]]

    for i in range(1, len(ponteiros)):
        atual = ponteiros[i]
        anterior = ponteiros[i - 1]

        if abs(atual['angulo'] - anterior['angulo']) <= tolerancia:
            clusters[-1].append(atual)
        else:
            clusters.append([atual])

    resumo = []

    for cluster in clusters:
        angulos = [c['angulo'] for c in cluster]
        tamanhos = [c['tamanho'] for c in cluster]

        resumo.append({
            'angulo': np.mean(angulos),
            'tamanho': np.max(tamanhos),
            'coords': max(cluster, key=lambda c: c['tamanho'])['coords']
        })

    resumo = sorted(resumo, key=lambda c: c['tamanho'], reverse=True)

    return resumo

# CONVERTER ÂNGULO -> TEMPO
def anguloParaTempo(angulo_hora, angulo_minuto):
    minutos = int(round(angulo_minuto / 6.0)) % 60
    horas = int(angulo_hora / 30.0)
    horas = horas % 12

    if horas == 0:
        horas = 12

    return horas, minutos

# LEITURA DO RELÓGIO
def lerRelogio(img):
    # resize
    img = resizeImagem(img, 500)
    output = img.copy()

    # preprocessamento
    gray = preprocessamentoCV(img)

    # detectar círculo
    circulo = detectarCirculo(gray)

    if circulo is None:
        print("Círculo não encontrado")
        return None

    cx, cy, raio = circulo

    # desenha círculo
    cv.circle(output, (cx, cy), raio, (0, 255, 0), 2)
    cv.circle(output, (cx, cy), 3, (255, 0, 0), -1)

    # bordas
    edges = cv.Canny(gray, 50, 150)

    # linhas
    linhas = detectarLinhas(edges)
    ponteiros = filtrarLinhas(linhas, cx, cy, raio)
    clusters = clusterizarPonteiros(ponteiros)

    if len(clusters) < 2:
        print("Não foi possível detectar ponteiros suficientes")
        return None

    # maior = minuto
    ponteiro_minuto = clusters[0]

    # segundo maior = hora
    ponteiro_hora = clusters[1]

    # desenhar ponteiros
    for p in [ponteiro_minuto, ponteiro_hora]:
        x1, y1, x2, y2 = p['coords']

        cv.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # tempo
    horas, minutos = anguloParaTempo(
        ponteiro_hora['angulo'],
        ponteiro_minuto['angulo']
    )

    print(f"Hora detectada: {horas:02d}:{minutos:02d}")

    # mostrar
    rgb = cv.cvtColor(output, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

    return horas, minutos

# MAIN
caminho = "imagem4.jpg"

img = cv.imread(caminho)

if img is None:
    print("Erro ao carregar imagem")
    exit()

# detectar relógio
bbox = detectarRelogio(img)

if bbox is None:
    print("Nenhum relógio detectado")
    exit()

x1, y1, x2, y2 = bbox

# desenhar bbox
img_box = img.copy()
cv.rectangle(img_box, (x1, y1), (x2, y2), (255, 0, 0), 3)

# crop do relógio
crop = img[y1:y2, x1:x2]

# visualizar detecção
rgb = cv.cvtColor(img_box, cv.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.axis('off')
plt.show()

# ler relógio
resultado = lerRelogio(crop)

if resultado is not None:
    horas, minutos = resultado
    print(f"Resultado final: {horas:02d}:{minutos:02d}")
