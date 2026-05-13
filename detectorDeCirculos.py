import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import math

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = "cpu"

if torch.cuda.is_available():
    model.cuda()
    device = torch.device("cuda")

def preprocess(img):
    imagem = img.copy()
    imagem = imagem.transpose(2, 0, 1)
    imagem /= 255.
    return imagem

def detect_clock(img):
    inp = [torch.from_numpy(preprocess(img)).float().to(device)]
    predict = model(inp)[0]

    boxes = predict['boxes'].detach()
    labels = predict['labels']
    scores = predict['scores']

    for i in range(len(labels)):
        label = labels[i].item()
        if label == 85:
            return boxes[i].cpu().numpy().round().astype(np.uint16)
    
    return None

def findLines(img):
    threshold = 50
    deg = np.pi / 180
    rad = 1
    min = 50
    max = 10

    return cv.HoughLinesP(img, rad, deg, threshold, minLineLength=min, maxLineGap=max)

def dist(a, b, c, d):
    return math.sqrt(math.pow(a - c, 2) + math.pow(b - d, 2))

def proximaCentro(a, b, c, d, x, y, raio):
    return dist(a, b, x, y) <= raio or dist(c, d, x, y) <= raio

def encontraPonteiros(linhas, x, y, raio):
    ponteiros = []

    for linha in linhas:
        for x1, y1, x2, y2 in linha:
            if proximaCentro(x1, y1, x2, y2, x, y, raio):
                if dist(x1, y1, x, y) > dist(x2, y2, x, y):
                    x1, y1, x2, y2 = x2, y2, x1, y1

                angulo = math.degrees(math.atan2(x2 - x1, y1 - y2))
                if angulo < 0: angulo += 360
                tamanho =  dist(x1, y1, x2, y2)

                ponteiros.append((angulo, tamanho))
    
    return ponteiros

def clusterLinhas(linhas):
    linhas.sort()
    max = 5
    clusters = [[linhas[0]]]

    for i in range(1, len(linhas)):
        if abs(linhas[i][0] - linhas[i-1][0]) <= max:
            clusters[len(clusters) - 1].append(linhas[i])
        else:
            clusters.append([linhas[i]])
    
    return clusters

def sumarizarClusters(clusters):
    summary = []

    for cluster in clusters:
        angles = np.array([angle for angle, length in cluster])
        lengths = np.array([length for angle, length in cluster])

        avg_angle = np.mean(angles)
        max_len = np.max(lengths)

        summary.append((max_len, avg_angle))

    summary.sort(reverse=True)

    return summary

def tempoAngulo(anguloH, anguloM):
    razaoH = anguloH / 360.
    razaoM = anguloM / 360.

    horas = razaoH * 12
    minutos = int(round(razaoM * 60)) % 60

    if abs(minutos - 60) < 5 or minutos < 5:
        horas = int(round(horas))
    else:
        horas = math.floor(horas)

    if horas == 0:
        horas = 12


    return horas, minutos

def tellTime(img):
    processed = preprocess(img)
    edges = cv.Canny(processed, 100, 200)
    linhas = findLines(edges)

    centroX = img.shape[1]/2
    centroY = img.shape[0]/2

    raio = 50

    ponteiros = encontraPonteiros(linhas, centroX, centroY, raio)

    clusters = clusterLinhas(ponteiros)
    sumario = sumarizarClusters(clusters)

    if len(sumario) == 1: sumario.append(sumario[0])

    horas, minutos = tempoAngulo(sumario[1][1], sumario[0][1])

    for linha in linhas:
        for x1, y1, x2, y2 in linha:
            if proximaCentro(x1, y1, x2, y2, centroX, centroY, raio):
                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv.imshow("lines", img)
    cv.imshow("edges", edges)

    return horas, minutos


