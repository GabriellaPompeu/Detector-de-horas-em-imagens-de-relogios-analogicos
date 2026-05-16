import os
from pathlib import Path
import cv2 as cv

Img = cv.typing.MatLike
PASTA_DESTINO = "resultados"


def listar_imagens(pasta: str) -> list[str]:
    '''Lista todos os arquivos de imagem de uma pasta.'''

    if not os.path.isdir(pasta):
        raise FileNotFoundError(f"A pasta '{pasta}' não existe.")

    extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    imagens = []

    for arquivo in Path(pasta).iterdir():
        if (arquivo.is_file() and arquivo.suffix.lower() in extensoes):
            imagens.append(str(arquivo.resolve()))

    return imagens


def salvar_imagem(img: Img, nome: str):
    # Criar a pasta se não existir
    os.makedirs(PASTA_DESTINO, exist_ok=True)

    # Monta um nome característico pros resultados
    nome_base = Path(nome).stem
    nick = f"{nome_base}_resultado.jpg"
    
    caminho_saida = os.path.join(PASTA_DESTINO, nick)

    # salvar resultado
    cv.imwrite(caminho_saida, img)

    print("Imagem salva")
