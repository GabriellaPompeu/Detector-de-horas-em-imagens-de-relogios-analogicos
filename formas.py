import math

import cv2 as cv
import numpy as np


class Circulo:
    def __init__(self, cx:float, cy:float, raio:float):
        self.cx = cx
        self.cy = cy
        self.raio = raio
    
    def desenhar(self, output, color:tuple[int], thickness: int):
        cv.circle(output, (self.cx, self.cy), self.raio, color, thickness)
    
    def desenhar_centro(self, output, color:tuple[int], thickness: int):
        cv.circle(output, (self.cx, self.cy), thickness, color, -1)


class Linha:
    def __init__(self, x1:float, y1:float, x2:float, y2:float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.comprimento = None
        self.angulo = None
    
    def get_comprimento(self) -> float:
        if not self.comprimento:
            self.comprimento = math.dist((self.x1, self.y1), (self.x2, self.y2))
        return self.comprimento

    def get_angulo(self) -> float:
        if not self.angulo:
            dx = self.x2 - self.x1
            dy = self.y1 - self.y2

            self.angulo = math.degrees(math.atan2(dy, dx))

            if self.angulo < 0:
                self.angulo += 360
            
        return self.angulo
    
    def trocar_pontos(self) -> None:
        self.x1, self.y1 = self.x2, self.y2
    
    def desenhar(self, output, color:tuple[int], thickness: int):
        cv.line(output, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), color, thickness)
    
    def ponta_mais_proxima(self, ponto:tuple[float]):
        if math.dist((self.x1, self.y1), ponto) < math.dist((self.x2, self.y2), ponto):
            return (self.x1, self.y1)
        return (self.x2, self.y2)

    def ponta_mais_distante(self, ponto:tuple[float]):
        if math.dist((self.x1, self.y1), ponto) > math.dist((self.x2, self.y2), ponto):
            return (self.x1, self.y1)
        return (self.x2, self.y2)


def linha_media(linhas: list[Linha]) -> Linha:
    p1 = [(linhas[0].x1, linhas[0].y1)]
    p2 = [(linhas[0].x2, linhas[0].y2)]
    for linha in linhas:
        d1 = math.dist((linha.x1, linha.y1), (p1[0][0], p1[0][1]))
        d2 = math.dist((linha.x2, linha.y2), (p1[0][0], p1[0][1]))
        if d1 < d2:
            p1.append((linha.x1, linha.y1))
            p2.append((linha.x2, linha.y2))
        else:
            p1.append((linha.x2, linha.y2))
            p2.append((linha.x1, linha.y1))

    x1 = np.mean([p[0] for p in p1])
    y1 = np.mean([p[1] for p in p1])
    x2 = np.mean([p[0] for p in p2])
    y2 = np.mean([p[1] for p in p2])

    return Linha(x1, y1, x2, y2)

        