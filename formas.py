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
        if self.comprimento is None: #não recalcula quando é zero
            self.comprimento = math.dist((self.x1, self.y1), (self.x2, self.y2))
        return self.comprimento

    def get_angulo(self) -> float:
        if self.angulo is None: #não recalcula quando é zero
            dx = self.x2 - self.x1
            dy = self.y1 - self.y2

            #self.angulo = math.degrees(math.atan2(dy, dx))
            ang = math.degrees(math.atan2(dy, dx))
            self.angulo = ang % 180
            #if self.angulo < 0:
            #    self.angulo += 360
            
        return self.angulo
    
    def trocar_pontos(self) -> None:
        self.x1, self.y1, self.x2, self.y2 = self.x2, self.y2, self.x1, self.y1
    
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
    
    def esticar_ate_ponto(self, ponto:tuple[float]):
        if math.dist((self.x1, self.y1), ponto) > math.dist((self.x2, self.y2), ponto):
            self.x2, self.y2 = ponto
        else:
            self.x1, self.y1 = ponto



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

    #x1 = np.mean([p[0] for p in p1])
    #y1 = np.mean([p[1] for p in p1])
    #x2 = np.mean([p[0] for p in p2])
    #y2 = np.mean([p[1] for p in p2])

    mx = np.mean([p1[0], p2[0]])
    my = np.mean([p1[1], p2[1]])
    p1f = max(p1, key=lambda p: math.dist(p, (mx, my)))
    p2f = max(p2, key=lambda p: math.dist(p, (mx, my)))

    return Linha(*p1f, *p2f)


class Relogio:
    def __init__(self, circulo: Circulo, ponteiro_m: Linha, ponteiro_h: Linha):
        self.circulo = circulo
        self.ponteiro_m = ponteiro_m
        self.ponteiro_h = ponteiro_h
        self.cx = circulo.cx
        self.cy = circulo.cy
    
    def _angulo_do_ponteiro(self, pont:Linha) -> float:
        '''Angulo baseado no centro do relógio e a ponta do ponteiro.'''
        p = pont.ponta_mais_distante((self.cx, self.cy))
        angulo = math.degrees(math.atan2(p[1] - self.cy, p[0] - self.cx)) + 90
        if angulo < 0:
            angulo += 360
        return angulo

    def angulo_do_ponteiro_minuto(self) -> float:
        return self._angulo_do_ponteiro(self.ponteiro_m)
    
    def angulo_do_ponteiro_hora(self) -> float:
        return self._angulo_do_ponteiro(self.ponteiro_h)

    def calcular_hora(self) -> tuple[int]:
        angulo_hora = self.angulo_do_ponteiro_hora()
        angulo_minuto = self.angulo_do_ponteiro_minuto()
        #print(angulo_hora, angulo_minuto)
        horas = angulo_hora / 30
        minutos = (angulo_minuto / 6) % 60
        print(f'{horas=}, {minutos=}')
        
        # se estiver muito perto de uma hora,
        # arredonda de forma inteligente baseado nos minutos
        if abs(horas - round(horas)) < .2:
            if minutos > 30:
                horas = (round(horas) - 1) % 12
            else:
                horas = round(horas)
        
        horas = int(horas)
        minutos = int(minutos)

        if horas == 0:
            horas = 12

        return horas, minutos

def diferenca_angulo(ang_1, ang_2):
    ang_1 %= 360
    ang_2 %= 360
    mi, ma = min(ang_1, ang_2), max(ang_1, ang_2)
    return min(ma - mi, 360 - (ma - mi))
