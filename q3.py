import numpy as np
import matplotlib.pyplot as plt


def plotaSolido(ax, pontos, arestas, color="g"):
    for aresta in arestas:
        x = [pontos[0][aresta[0]], pontos[0][aresta[1]]]
        y = [pontos[1][aresta[0]], pontos[1][aresta[1]]]
        z = [pontos[2][aresta[0]], pontos[2][aresta[1]]]
        ax.plot(x, y, z, color)


# def rotacionar(angle_x, angle_y, angle_z, x, y, z):
#     angle_x = np.radians(angle_x)
#     angle_y = np.radians(angle_y)
#     angle_z = np.radians(angle_z)

#     rotation_x = np.array(
#         [
#             [1, 0, 0],
#             [0, np.cos(angle_x), -np.sin(angle_x)],
#             [0, np.sin(angle_x), np.cos(angle_x)],
#         ]
#     )

#     rotation_y = np.array(
#         [
#             [np.cos(angle_y), 0, np.sin(angle_y)],
#             [0, 1, 0],
#             [-np.sin(angle_y), 0, np.cos(angle_y)],
#         ]
#     )

#     rotation_z = np.array(
#         [
#             [np.cos(angle_z), -np.sin(angle_z), 0],
#             [np.sin(angle_z), np.cos(angle_z), 0],
#             [0, 0, 1],
#         ]
#     )
#     rotated_points = []
#     for i in range(len(x)):
#         point = np.array([x[i], y[i], z[i]])
#         rotated_point = np.dot(
#             rotation_x, np.dot(rotation_y, np.dot(rotation_z, point))
#         )
#         rotated_points.append(rotated_point)

#     return zip(*rotated_points)


# def escalar(sx, sy, sz, x, y, z, ponto_inicial):
#     # Cria matriz de escala
#     matriz_escala = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
#     pontos_scaled = matriz_escala.dot([x, y, z])

#     # Ajustar a posição dos pontos escalados
#     pontos_scaled[0] += ponto_inicial[0] * (1 - sx)
#     pontos_scaled[1] += ponto_inicial[1] * (1 - sy)
#     pontos_scaled[2] += ponto_inicial[2] * (1 - sz)
#     return pontos_scaled[0], pontos_scaled[1], pontos_scaled[2]


# def deslocar(dx, dy, dz, x, y, z):
#     T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])
#     complete_row = np.full((1, len(x)), 1)[0]
#     new_matrix = T.dot([x, y, z, complete_row])
#     x = new_matrix[0]
#     y = new_matrix[1]
#     z = new_matrix[2]
#     return x, y, z


class Cubo:
    def __init__(self, raio, ponto_inicial):
        self.raio = raio
        self.ponto_inicial = ponto_inicial
        self.x, self.y, self.z, self.arestas = [], [], [], []
        self.base = [
            self.ponto_inicial,
            (
                self.ponto_inicial[0] + self.raio,
                self.ponto_inicial[1],
                self.ponto_inicial[2],
            ),
            (
                self.ponto_inicial[0] + self.raio,
                self.ponto_inicial[1] + self.raio,
                self.ponto_inicial[2],
            ),
            (
                self.ponto_inicial[0],
                self.ponto_inicial[1] + self.raio,
                self.ponto_inicial[2],
            ),
        ]

    def formarBases(self):
        contador = 0
        for index in range(len(self.x)):
            if contador != 3:
                contador += 1
                self.arestas.append([index, index + 1])
            else:
                self.arestas.append([index, index - 3])
                contador = 0

    def formarArestasVerticais(self):
        for index in range(len(self.x)):
            if index < (len(self.x) - 4):
                self.arestas.append([index, index + 4])

    def generate_cubo(self):
        for bases in range(2):
            for j in range(len(self.base)):
                self.x.append(self.base[j][0])
                self.y.append(self.base[j][1])
                self.z.append(self.base[j][2] + bases * self.raio)

    # def rotacionar_cubo(self, angle_x, angle_y, angle_z):
    #     self.x, self.y, self.z = rotacionar(
    #         angle_x, angle_y, angle_z, self.x, self.y, self.z
    #     )

    # def deslocar_cubo(self, dx, dy, dz):
    #     self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    # def escalar_cubo(self, sx, sy, sz):
    #     self.x, self.y, self.z = escalar(
    #         sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
    #     )

    def gera_cubo(self):
        self.generate_cubo()
        self.formarBases()
        self.formarArestasVerticais()

    def plota_cubo(self, ax):
        plotaSolido(ax, [self.x, self.y, self.z], self.arestas)


class Cone:
    def __init__(self, radius, height, num_slices, ponto_inicial):
        self.radius = radius
        self.height = height
        self.num_slices = num_slices
        self.ponto_inicial = ponto_inicial
        self.pontosX, self.pontosY, self.pontosZ, self.arestas = [], [], [], []

    def generate_cone(self):
        theta = np.linspace(0, 2 * np.pi, self.num_slices)
        z = np.linspace(0, self.height, self.num_slices)

        x = np.empty((self.num_slices, self.num_slices))
        y = np.empty((self.num_slices, self.num_slices))
        z_2d = np.empty((self.num_slices, self.num_slices))
        for i in range(self.num_slices):
            x[i] = self.ponto_inicial[0] + self.radius * (
                1 - z[i] / self.height
            ) * np.cos(theta)
            y[i] = self.ponto_inicial[1] + self.radius * (
                1 - z[i] / self.height
            ) * np.sin(theta)
            z_2d[i] = self.ponto_inicial[2] + z[i]

        return x, y, z_2d

    def gerar_cone(self):
        contador = 0
        x, y, z = self.generate_cone()
        for i in range(self.num_slices):
            for j in range(len(x[i])):
                self.pontosX.append(x[i][j])
                self.pontosY.append(y[i][j])
                self.pontosZ.append(z[i][j])
                if j < (len(x[i]) - 1):
                    self.arestas.append([contador, contador + 1])
                contador += 1
            for circles in range(len(x[i])):
                self.arestas.append([i, i + self.num_slices * circles])

    # def rotacionar_cone(self, angle_x, angle_y, angle_z):
    #     self.pontosX, self.pontosY, self.pontosZ = rotacionar(
    #         angle_x, angle_y, angle_z, self.pontosX, self.pontosY, self.pontosZ
    #     )

    # def deslocar_cone(self, dx, dy, dz):
    #     self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    # def escalar_cone(self, sx, sy, sz):
    #     self.x, self.y, self.z = escalar(
    #         sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
    #     )

    def plota_cone(self, ax):
        # Plotar o cone
        plotaSolido(ax, [self.pontosX, self.pontosY, self.pontosZ], self.arestas)


class Cilindro:
    def __init__(self, raio, altura, ponto_inicial):
        self.raio = raio
        self.altura = altura
        self.ponto_inicial = ponto_inicial
        self.x, self.y, self.z, self.arestas = [], [], [], []

    def gera_cilindro(self):
        # Configurando os parâmetros do cilindro
        resolucao = (
            15  # Resolução do cilindro (quanto maior, mais suave será a superfície)
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

        cont = 0
        # Percorrer as células da grade e plotar os segmentos de linha
        for i in range(num_rows - 1):
            for j in range(num_cols - 1):
                # Obter as coordenadas dos vértices da célula atual
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

                if cont < resolucao - 1:
                    self.arestas.append([posicaov1, posicaov2])  # H
                    cont += 1

                self.arestas.append([posicaov2, posicaov3])  # V
                self.arestas.append([posicaov3, posicaov4])  # H

    # def escalar_cilindro(self, sx, sy, sz):
    #     self.x, self.y, self.z = escalar(
    #         sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
    #     )

    # def deslocar_cilindro(self, dx, dy, dz):
    #     self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    # def rotacionar_cilindro(self, angle_x, angle_y, angle_z):
    #     self.x, self.y, self.z = rotacionar(
    #         angle_x, angle_y, angle_z, self.x, self.y, self.z
    #     )

    def plota_cilindro(self, ax):
        plotaSolido(ax, [self.x, self.y, self.z], self.arestas)


class Tronco_piramide:
    def __init__(self, aresta_inferior, aresta_superior, altura, ponto_inicial):
        self.aresta_superior = aresta_superior
        self.aresta_inferior = aresta_inferior
        self.altura = altura
        self.ponto_inicial = ponto_inicial
        self.x, self.y, self.z, self.arestas = [], [], [], []
        self.baseInferior = np.array(
            [
                (self.ponto_inicial[0], self.ponto_inicial[1], self.ponto_inicial[2]),
                (
                    self.ponto_inicial[0] + self.aresta_inferior,
                    self.ponto_inicial[1],
                    self.ponto_inicial[2],
                ),
                (
                    self.ponto_inicial[0] + self.aresta_inferior,
                    self.ponto_inicial[1] + self.aresta_inferior,
                    self.ponto_inicial[2],
                ),
                (
                    self.ponto_inicial[0],
                    self.ponto_inicial[1] + self.aresta_inferior,
                    self.ponto_inicial[2],
                ),
            ]
        )
        self.deslocamento = (self.aresta_inferior - self.aresta_superior) / 2
        self.baseSuperior = np.array(
            [
                (
                    self.ponto_inicial[0] + self.deslocamento,
                    self.ponto_inicial[1] + self.deslocamento,
                    self.ponto_inicial[2],
                ),
                (
                    self.ponto_inicial[0] + self.aresta_superior + self.deslocamento,
                    self.ponto_inicial[1] + self.deslocamento,
                    self.ponto_inicial[2],
                ),
                (
                    self.ponto_inicial[0] + self.aresta_superior + self.deslocamento,
                    self.ponto_inicial[1] + self.aresta_superior + self.deslocamento,
                    self.ponto_inicial[2],
                ),
                (
                    self.ponto_inicial[0] + self.deslocamento,
                    self.ponto_inicial[1] + self.aresta_superior + self.deslocamento,
                    self.ponto_inicial[2],
                ),
            ]
        )
        self.bases = [self.baseInferior, self.baseSuperior]

    def formarBases(self):
        contador = 0
        for index in range(len(self.x)):
            if contador != 3:
                contador += 1
                self.arestas.append([index, index + 1])
            else:
                self.arestas.append([index, index - 3])
                contador = 0

    def formarArestasVerticais(self):
        for index in range(len(self.x)):
            if index < (len(self.x) - 4):
                self.arestas.append([index, index + 4])

    def gera_tronco(self):
        # Inicializar as listas x, y, z antes de formar as bases e as arestas
        self.x = [vertex[0] for base in self.bases for vertex in base]
        self.y = [vertex[1] for base in self.bases for vertex in base]
        self.z = [
            vertex[2] + self.altura * index
            for index, base in enumerate(self.bases)
            for vertex in base
        ]
        self.formarBases()
        self.formarArestasVerticais()

    # def deslocar_tronco(self, dx, dy, dz):
    #     self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    # def rotacionar_tronco(self, angle_x, angle_y, angle_z):
    #     self.x, self.y, self.z = rotacionar(
    #         angle_x, angle_y, angle_z, self.x, self.y, self.z
    #     )

    # def escalar_tronco(self, sx, sy, sz):
    #     self.x, self.y, self.z = escalar(
    #         sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
    #     )

    def plota_tronco(self, ax):
        plotaSolido(ax, [self.x, self.y, self.z], self.arestas)


class Esfera:
    def __init__(self, raio, ponto_inicial):
        self.raio = raio
        self.ponto_inicial = ponto_inicial
        self.x, self.y, self.z, self.arestas = [], [], [], []

    def gera_esfera(self):
        num_pontos = 15
        t_incremento = 2 * np.pi / num_pontos
        p_incremento = np.pi / num_pontos
        theta = 0.0
        phi = 0.0

        for _ in range(num_pontos + 1):
            phi = 0.0
            for _ in range(num_pontos + 1):
                self.x.append(
                    self.ponto_inicial[0] + self.raio * np.cos(theta) * np.sin(phi)
                )
                self.y.append(
                    self.ponto_inicial[1] + self.raio * np.sin(theta) * np.sin(phi)
                )
                self.z.append(self.ponto_inicial[2] + self.raio * np.cos(phi))

                phi += p_incremento

            theta += t_incremento

        for i in range(num_pontos + 1):
            for j in range(num_pontos):
                self.arestas.append(
                    [i * (num_pontos + 1) + j, i * (num_pontos + 1) + j + 1]
                )
                self.arestas.append(
                    [j * (num_pontos + 1) + i, (j + 1) * (num_pontos + 1) + i]
                )

    # def escalar_esfera(self, sx, sy, sz):
    #     self.x, self.y, self.z = escalar(
    #         sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
    #     )

    # def deslocar_esfera(self, dx, dy, dz):
    #     self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    # def rotacionar_esfera(self, angle_x, angle_y, angle_z):
    #     self.x, self.y, self.z = rotacionar(
    #         angle_x, angle_y, angle_z, self.x, self.y, self.z
    #     )

    def plota_esfera(self, ax):
        plotaSolido(ax, [self.x, self.y, self.z], self.arestas)


# centro de massa dos objetos(média vértices)
def calculateCenterMass(vertexList):
    return np.mean(vertexList, axis=0)


def convertWorldToCamera(vertexList, U, V, N, eye):
    RTMatrix = np.array(
        [
            [U[0], U[1], U[2], -np.dot(eye, U)],
            [V[0], V[1], V[2], -np.dot(eye, V)],
            [N[0], N[1], N[2], -np.dot(eye, N)],
            [0, 0, 0, 1],
        ]
    )
    cameraVertexList = np.dot(
        RTMatrix, np.concatenate([vertexList.T, np.ones((1, vertexList.shape[0]))])
    )
    cameraVertexList = cameraVertexList[:3].T
    return cameraVertexList  # isso aqui é o solido transformado (já foi rotacionado e transladado)


# ENTRADA
cubo = Cubo(4, [-6, -8, -7])
cubo.gera_cubo()
cubeVertexList = np.array([cubo.x, cubo.y, cubo.z]).T

cone = Cone(1, 2, 5, [8, 6, 4])
cone.gerar_cone()
coneVertexList = np.array([cone.pontosX, cone.pontosY, cone.pontosZ]).T

cilindro = Cilindro(1, 2, [2, 4, 3])
cilindro.gera_cilindro()
cilindroVertexList = np.array([cilindro.x, cilindro.y, cilindro.z]).T

tronco = Tronco_piramide(2, 1, 2, [-9, -3, -8])
tronco.gera_tronco()
troncoVertexList = np.array([tronco.x, tronco.y, tronco.z]).T

esfera = Esfera(1, [7, -2, -6])
esfera.gera_esfera()
esferaVertexList = np.array([esfera.x, esfera.y, esfera.z]).T

# Calculando os centros de massa das formas geométricas
cubeCenterMass = calculateCenterMass(cubeVertexList)
coneCenterMass = calculateCenterMass(coneVertexList)
cilindroCenterMass = calculateCenterMass(cilindroVertexList)
troncoCenterMass = calculateCenterMass(troncoVertexList)
esferaCenterMass = calculateCenterMass(esferaVertexList)

# calculando a média de todos os centros para apontar a câmera
soma = (
    cubeCenterMass
    + coneCenterMass
    + cilindroCenterMass
    + troncoCenterMass
    + esferaCenterMass
) / 5

# Posição da câmera
eye = np.array([-7, -1, 6])
# -7,-1,6
# 1,-4,2
# -2, 3, -5

# Calculando os vetores N, U e V da câmera
at = np.array(soma)
n = at - eye
aux = np.array([0, 1, 0])
v = np.cross(aux, n)
u = np.cross(v, n)
N = n / np.linalg.norm(n)
V = v / np.linalg.norm(v)
U = u / np.linalg.norm(u)

# Criação da figura e do subplot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Desenhando o cubo no S.C.C
cubeVertexListCamera = convertWorldToCamera(cubeVertexList, U, V, N, eye)
cubo.plota_cubo(ax)

coneVertexListCamera = convertWorldToCamera(coneVertexList, U, V, N, eye)
cone.plota_cone(ax)

cilindroVertexListCamera = convertWorldToCamera(cilindroVertexList, U, V, N, eye)
cilindro.plota_cilindro(ax)

troncoVertexListCamera = convertWorldToCamera(troncoVertexList, U, V, N, eye)
tronco.plota_tronco(ax)

esferaVertexListCamera = convertWorldToCamera(esferaVertexList, U, V, N, eye)
esfera.plota_esfera(ax)

# Cube.drawCube(ax, cubeEdgeList, 'b')

plotaSolido(ax, cubeVertexListCamera.T, cubo.arestas, color="b")
plotaSolido(ax, coneVertexListCamera.T, cone.arestas, color="b")
plotaSolido(ax, cilindroVertexListCamera.T, cilindro.arestas, color="b")
plotaSolido(ax, troncoVertexListCamera.T, tronco.arestas, color="b")
plotaSolido(ax, esferaVertexListCamera.T, esfera.arestas, color="b")


# Plot do ponto de vista da câmera
ax.scatter(eye[0], eye[1], eye[2], color="r", label="eye")

# Média dos pontos
ax.scatter(soma[0], soma[1], soma[2], color="y", label="at")

# Ponto Origem do Mundo
ax.scatter(0, 0, 0, color="m", label="origem")

# Configurações do gráfico
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# Exibição do gráfico
plt.show()
