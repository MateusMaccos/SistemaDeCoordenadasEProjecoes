import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def rotacionar(angle_x, angle_y, angle_z, x, y, z):
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)

    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )

    rotation_y = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )

    rotation_z = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )
    rotated_points = []
    for i in range(len(x)):
        point = np.array([x[i], y[i], z[i]])
        rotated_point = np.dot(
            rotation_x, np.dot(rotation_y, np.dot(rotation_z, point))
        )
        rotated_points.append(rotated_point)

    return zip(*rotated_points)


def escalar(sx, sy, sz, x, y, z, ponto_inicial):
    # Cria matriz de escala
    matriz_escala = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
    pontos_scaled = matriz_escala.dot([x, y, z])

    # Ajustar a posição dos pontos escalados
    pontos_scaled[0] += ponto_inicial[0] * (1 - sx)
    pontos_scaled[1] += ponto_inicial[1] * (1 - sy)
    pontos_scaled[2] += ponto_inicial[2] * (1 - sz)
    return pontos_scaled[0], pontos_scaled[1], pontos_scaled[2]


def deslocar(dx, dy, dz, x, y, z):
    T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])
    complete_row = np.full((1, len(x)), 1)[0]
    new_matrix = T.dot([x, y, z, complete_row])
    x = new_matrix[0]
    y = new_matrix[1]
    z = new_matrix[2]
    return x, y, z


def plotaSolido(pontos, arestas):
    for aresta in arestas:
        x = [pontos[0][aresta[0]], pontos[0][aresta[1]]]
        y = [pontos[1][aresta[0]], pontos[1][aresta[1]]]
        z = [pontos[2][aresta[0]], pontos[2][aresta[1]]]
        ax.plot(x, y, z, "b")


class Cilindro:
    def __init__(self, raio, altura, ponto_inicial):
        self.raio = raio
        self.altura = altura
        self.ponto_inicial = ponto_inicial
        self.x, self.y, self.z, self.arestas = [], [], [], []

    def gera_cilindro(self):
        # Configurando os parâmetros do cilindro
        resolucao = (
            20  # Resolução do cilindro (quanto maior, mais suave será a superfície)
        )

        # Criando os pontos para a superfície do cilindro
        theta = np.linspace(0, 2 * np.pi, resolucao)
        z_var = np.linspace(
            self.ponto_inicial[2], self.ponto_inicial[2] + self.altura, resolucao
        )
        theta, z_var = np.meshgrid(theta, z_var)
        x_var = self.ponto_inicial[0] + self.raio * np.cos(theta)
        y_var = self.ponto_inicial[1] + self.raio * np.sin(theta)

        # Obter as dimensões dos dados
        num_rows, num_cols = x_var.shape

        # Percorrer as células da grade e plotar os segmentos de linha
        for i in range(num_rows - 1):
            for j in range(num_cols - 1):
                # Obter as coordenadas dos vértices da célula atual(Faz o retângulo e interliga)
                v1 = [x_var[i, j], y_var[i, j], z_var[i, j]]
                v2 = [x_var[i, j + 1], y_var[i, j + 1], z_var[i, j + 1]]
                v3 = [x_var[i + 1, j + 1], y_var[i + 1, j + 1], z_var[i + 1, j + 1]]
                v4 = [x_var[i + 1, j], y_var[i + 1, j], z_var[i + 1, j]]

                self.x.extend((v1[0], v2[0], v3[0], v4[0]))
                self.y.extend((v1[1], v2[1], v3[1], v4[1]))
                self.z.extend((v1[2], v2[2], v3[2], v4[2]))

                posicaov1 = len(self.x) - 4
                posicaov2 = len(self.x) - 3
                posicaov3 = len(self.x) - 2
                posicaov4 = len(self.x) - 1

                self.arestas.append([posicaov1, posicaov2])  # H
                self.arestas.append([posicaov2, posicaov3])  # V
                self.arestas.append([posicaov3, posicaov4])  # H

    def escalar_cilindro(self, sx, sy, sz):
        self.x, self.y, self.z = escalar(
            sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
        )

    def deslocar_cilindro(self, dx, dy, dz):
        self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    def rotacionar_cilindro(self, angle_x, angle_y, angle_z):
        self.x, self.y, self.z = rotacionar(
            angle_x, angle_y, angle_z, self.x, self.y, self.z
        )

    def plota_cilindro(self):
        plotaSolido([self.x, self.y, self.z], self.arestas)


# Cilindro  #ESCALADO (n do jeito que é pra ser)
raio_cilindro = 1
altura_cilindro = 2
ponto_inicial_cilindro = [2, 4, 3]
cilindro = Cilindro(raio_cilindro, altura_cilindro, ponto_inicial_cilindro)
cilindro.gera_cilindro()
cilindro.plota_cilindro()

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
