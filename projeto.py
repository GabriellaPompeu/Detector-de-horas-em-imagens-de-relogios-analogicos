from ultralytics import YOLO
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# MODELO YOLO SEGMENTATION
model = YOLO("yolov8n-seg.pt")

# PREPROCESSAMENTO
def preprocessamentoCV(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # melhora contraste
    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # preserva bordas
    gray = cv.bilateralFilter(gray, 7, 50, 50)

    return gray

# RESIZE
def resizeImagem(img, largura=500):
    h, w = img.shape[:2]

    escala = largura / w

    nova_altura = int(h * escala)

    img = cv.resize(img, (largura, nova_altura))

    return img

# DETECTAR RELÓGIO COM SEGMENTAÇÃO
def detectarRelogio(img):
    results = model(img, verbose=False)[0]

    if results.masks is None:
        return None

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

        if score < 0.5:
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
            mask = cv.resize(
                mask,
                (img.shape[1], img.shape[0])
            )

            mask_crop = mask[y1:y2, x1:x2]
            melhor_crop = crop
            melhor_mask = mask_crop
            melhor_box = (x1, y1, x2, y2)

    if melhor_crop is None:
        return None

    return melhor_crop, melhor_mask, melhor_box

# REMOVER FUNDO
def removerFundo(img, mask):
    mask_bin = (mask > 0.5).astype(np.uint8) * 255

    # suavizar máscara
    mask_bin = cv.GaussianBlur(mask_bin, (7,7), 0)

    resultado = cv.bitwise_and(
        img,
        img,
        mask=mask_bin
    )

    return resultado, mask_bin

# DETECTAR CÍRCULO
def detectarCirculo(gray):
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
    x, y, r = max(circles, key=lambda c: c[2])

    return x, y, r

# DISTÂNCIA
def distancia(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )

# DETECTAR LINHAS
def detectarLinhas(edges):
    linhas = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=40,
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

        # uma ponta precisa tocar centro
        if d1 > raio * 0.25 and d2 > raio * 0.25:
            continue

        # reorganiza
        if d1 > d2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        tamanho = distancia(x1, y1, x2, y2)
        # ponta da linha deve ficar dentro do relógio
        if distancia(x2, y2, cx, cy) > raio * 1.05:
            continue

        # remove linhas pequenas
        if tamanho < raio * 0.35:
            continue

        dx = x2 - x1
        dy = y1 - y2

        angulo = math.degrees(
            math.atan2(dx, dy)
        )

        if angulo < 0:
            angulo += 360

        ponteiros.append({
            "angulo": angulo,
            "tamanho": tamanho,
            "coords": (x1, y1, x2, y2)
        })

    return ponteiros

# CLUSTERIZAÇÃO
def clusterizarPonteiros(ponteiros, raio, tolerancia=8):

    if len(ponteiros) == 0:
        return []

    ponteiros = sorted(
        ponteiros,
        key=lambda p: p["angulo"]
    )

    clusters = [[ponteiros[0]]]

    for i in range(1, len(ponteiros)):

        atual = ponteiros[i]
        anterior = ponteiros[i - 1]

        if abs(
            atual["angulo"] -
            anterior["angulo"]
        ) <= tolerancia:

            clusters[-1].append(atual)

        else:
            clusters.append([atual])

    resumo = []

    for cluster in clusters:

        angulos = [
            c["angulo"]
            for c in cluster
        ]

        tamanhos = [
            c["tamanho"]
            for c in cluster
        ]

        coords = max(
            cluster,
            key=lambda c: c["tamanho"]
        )["coords"]

        tamanho_medio = np.mean(tamanhos)

        resumo.append({

            "angulo": np.mean(angulos),

            "tamanho": tamanho_medio,

            "coords": coords,

            "linhas": len(cluster),

            "razao_raio": tamanho_medio / raio
        })

    # FILTRO DO PONTEIRO DE SEGUNDOS
    candidatos = []

    for r in resumo:

        razao = r["razao_raio"]

        # remove ponteiros MUITO longos
        # normalmente segundos
        if razao > 0.92:
            continue

        # remove linhas pequenas
        if razao < 0.35:
            continue

        candidatos.append(r)

    if len(candidatos) < 2:
        return []

    # ESCOLHA DOS PONTEIROS
    candidatos = sorted(
        candidatos,
        key=lambda c: c["tamanho"],
        reverse=True
    )

    # minuto = maior restante
    ponteiro_minuto = candidatos[0]

    # hora = menor que minuto
    ponteiro_hora = None

    for c in candidatos[1:]:

        # ponteiro de hora deve ser
        # significativamente menor
        if c["tamanho"] < ponteiro_minuto["tamanho"] * 0.80:

            ponteiro_hora = c
            break

    # fallback
    if ponteiro_hora is None:

        ponteiro_hora = candidatos[1]

    return [
        ponteiro_minuto,
        ponteiro_hora
    ]

# ÂNGULO -> TEMPO
def anguloParaTempo(angulo_hora, angulo_minuto):
    minutos = int(
        round(angulo_minuto / 6.0)
    ) % 60

    horas = int(
        angulo_hora / 30.0
    ) % 12

    if horas == 0:
        horas = 12

    return horas, minutos

# LEITURA DO RELÓGIO
def lerRelogio(img, mask_segmentacao):
    img = resizeImagem(img, 500)

    # resize máscara
    mask_segmentacao = cv.resize(
        mask_segmentacao,
        (img.shape[1], img.shape[0])
    )

    output = img.copy()

    # remove fundo
    sem_fundo, mask = removerFundo(
        img,
        mask_segmentacao
    )

    # preprocessamento
    gray = preprocessamentoCV(sem_fundo)

    # detectar círculo
    circulo = detectarCirculo(gray)

    if circulo is None:
        print("Círculo não encontrado")
        return None

    cx, cy, raio = circulo

    # desenha círculo
    cv.circle(
        output,
        (cx, cy),
        raio,
        (0, 255, 0),
        2
    )

    cv.circle(
        output,
        (cx, cy),
        3,
        (255, 0, 0),
        -1
    )

    # máscara circular interna
    mask_clock = np.zeros_like(gray)

    cv.circle(
        mask_clock,
        (cx, cy),
        int(raio * 0.92),
        255,
        -1
    )

    gray = cv.bitwise_and(
        gray,
        gray,
        mask=mask_clock
    )

    # bordas
    edges = cv.Canny(gray, 70, 180)
    kernel = np.ones((3,3), np.uint8)
    edges = cv.morphologyEx(
        edges,
        cv.MORPH_CLOSE,
        kernel
    )

    # linhas
    linhas = detectarLinhas(edges)
    ponteiros = filtrarLinhas(
        linhas,
        cx,
        cy,
        raio
    )

    clusters = clusterizarPonteiros(
        ponteiros,
        raio
    )

    if len(clusters) < 2:
        print("Ponteiros insuficientes")
        return None

    # maior = minuto
    ponteiro_minuto = clusters[0]

    # segundo maior = hora
    ponteiro_hora = clusters[1]

    # desenhar ponteiros
    cores = [
        (0, 0, 255),
        (255, 0, 0)
    ]

    for i, p in enumerate([
        ponteiro_minuto,
        ponteiro_hora
    ]):

        x1, y1, x2, y2 = p["coords"]

        cv.line(
            output,
            (x1, y1),
            (x2, y2),
            cores[i],
            4
        )

    # tempo
    horas, minutos = anguloParaTempo(
        ponteiro_hora["angulo"],
        ponteiro_minuto["angulo"]
    )

    texto = f"{horas:02d}:{minutos:02d}"

    print(f"Hora detectada: {texto}")

    # desenhar texto
    cv.rectangle(
        output,
        (10, 10),
        (220, 70),
        (255,255,255),
        -1
    )

    cv.putText(
        output,
        texto,
        (20, 55),
        cv.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0,0,0),
        3
    )

    nick = caminho.replace(".jpg", "Resultado.jpg")
    # salvar resultado
    cv.imwrite(
        nick,
        output
    )

    print("Imagem salva")

    # mostrar
    rgb = cv.cvtColor(
        output,
        cv.COLOR_BGR2RGB
    )

    plt.figure(figsize=(8,8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(f"Hora predita: {texto}")
    plt.show()

    return horas, minutos

# MAIN
caminho = "imagem1.jpg"
img = cv.imread(caminho)

if img is None:
    print("Erro ao carregar imagem")
    exit()

resultado = detectarRelogio(img)

if resultado is None:
    print("Nenhum relógio detectado")
    exit()

crop, mask, bbox = resultado

x1, y1, x2, y2 = bbox

# visualizar detecção
img_box = img.copy()

cv.rectangle(
    img_box,
    (x1, y1),
    (x2, y2),
    (255,0,0),
    3
)

rgb = cv.cvtColor(
    img_box,
    cv.COLOR_BGR2RGB
)

plt.figure(figsize=(8,8))
plt.imshow(rgb)
plt.axis("off")
plt.title("Relógio detectado")
plt.show()

# ler relógio
resultado = lerRelogio(
    crop,
    mask
)

if resultado is not None:
    horas, minutos = resultado
    print(
        f"Resultado final: "
        f"{horas:02d}:{minutos:02d}"
    )