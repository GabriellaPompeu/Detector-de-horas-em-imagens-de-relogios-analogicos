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

    def _angulos_para_hora_minuto(self, ang_h: float, ang_m: float) -> tuple:
        '''Converte dois ângulos (hora, minuto) em (h, min) inteiros.'''
        minutos = int((ang_m / 6) % 60)
        horas_raw = ang_h / 30
        # ajuste fino: o ponteiro de horas avança ~0.5° por minuto
        # então a posição esperada é (H + min/60) * 30°
        # arredondamos quando estamos muito perto de uma hora cheia
        if abs(horas_raw - round(horas_raw)) < 0.1:
            horas_int = round(horas_raw) - 1 if minutos > 30 else round(horas_raw)
        else:
            horas_int = int(horas_raw)
        if horas_int == 0:
            horas_int = 12
        return horas_int, minutos

    def _erro_consistencia(self, ang_h: float, ang_m: float) -> float:
        '''Mede quão incoerente é a atribuição hora=ang_h, minuto=ang_m.
        O ponteiro de horas deve estar em (H + min/60) * 30 graus.
        Retorna o erro angular em graus (menor = mais coerente).'''
        h, m = self._angulos_para_hora_minuto(ang_h, ang_m)
        angulo_esperado_hora = ((h % 12) + m / 60) * 30  # 0..360
        diff = abs(ang_h - angulo_esperado_hora)
        if diff > 180:
            diff = 360 - diff
        return diff

    def calcular_hora(self) -> tuple[int]:
        ang_h = self._angulo_do_ponteiro(self.ponteiro_h)
        ang_m = self._angulo_do_ponteiro(self.ponteiro_m)

        # Testa as duas atribuições possíveis e escolhe a mais coerente.
        # Isso corrige o caso em que comprimentos parecidos fazem a
        # classificação hora/minuto ser atribuída ao ponteiro errado.
        erro_normal  = self._erro_consistencia(ang_h, ang_m)
        erro_trocado = self._erro_consistencia(ang_m, ang_h)

        if erro_trocado < erro_normal:
            # a atribuição invertida é mais coerente — usar ao contrário
            print(f"  Ponteiros trocados detectados (err_normal={erro_normal:.1f}° > "
                  f"err_trocado={erro_trocado:.1f}°). Corrigindo.")
            ang_h, ang_m = ang_m, ang_h

        return self._angulos_para_hora_minuto(ang_h, ang_m)
