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
        rotated_point = np.dot(rotation_x, np.dot(rotation_y, np.dot(rotation_z, point)))
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


def plotaSolido(pontos, arestas, cor = 'b'):
    for aresta in arestas:
        x = [pontos[0][aresta[0]], pontos[0][aresta[1]]]
        y = [pontos[1][aresta[0]], pontos[1][aresta[1]]]
        z = [pontos[2][aresta[0]], pontos[2][aresta[1]]]
        ax.plot(x, y, z, cor)


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

    def rotacionar_cone(self, angle_x, angle_y, angle_z):
        self.pontosX, self.pontosY, self.pontosZ = rotacionar(
            angle_x, angle_y, angle_z, self.pontosX, self.pontosY, self.pontosZ
        )

    def deslocar_cone(self, dx, dy, dz):
        self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    def escalar_cone(self, sx, sy, sz):
        self.x, self.y, self.z = escalar(
            sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
        )

    def plota_cone(self, cor = 'b'):
        # Plotar o cone
        plotaSolido([self.pontosX, self.pontosY, self.pontosZ], self.arestas, cor)


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

    def rotacionar_cubo(self, angle_x, angle_y, angle_z):
        self.x, self.y, self.z = rotacionar(
            angle_x, angle_y, angle_z, self.x, self.y, self.z
        )

    def deslocar_cubo(self, dx, dy, dz):
        self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    def escalar_cubo(self, sx, sy, sz):
        self.x, self.y, self.z = escalar(
            sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
        )

    def gera_cubo(self):
        self.generate_cubo()
        self.formarBases()
        self.formarArestasVerticais()

    def plota_cubo(self, cor = 'b'):
        
        plotaSolido([self.x, self.y, self.z], self.arestas, cor)


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

    def deslocar_tronco(self, dx, dy, dz):
        self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    def rotacionar_tronco(self, angle_x, angle_y, angle_z):
        self.x, self.y, self.z = rotacionar(
            angle_x, angle_y, angle_z, self.x, self.y, self.z
        )

    def escalar_tronco(self, sx, sy, sz):
        self.x, self.y, self.z = escalar(
            sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
        )

    def plota_tronco(self, cor = 'blue'):
        plotaSolido([self.x, self.y, self.z], self.arestas, cor)


class Esfera:
    def __init__(self, raio, ponto_inicial):
        self.raio = raio
        self.ponto_inicial = ponto_inicial
        self.x, self.y, self.z, self.arestas = [], [], [], []

    def gera_esfera(self):
        num_pontos = 10
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

    def escalar_esfera(self, sx, sy, sz):
        self.x, self.y, self.z = escalar(
            sx, sy, sz, self.x, self.y, self.z, self.ponto_inicial
        )

    def deslocar_esfera(self, dx, dy, dz):
        self.x, self.y, self.z = deslocar(dx, dy, dz, self.x, self.y, self.z)

    def rotacionar_esfera(self, angle_x, angle_y, angle_z):
        self.x, self.y, self.z = rotacionar(
            angle_x, angle_y, angle_z, self.x, self.y, self.z
        )

    def plota_esfera(self, cor = "b"):
        plotaSolido([self.x, self.y, self.z], self.arestas, cor)


class Cilindro:
    def __init__(self, raio, altura, ponto_inicial):
        self.raio = raio
        self.altura = altura
        self.ponto_inicial = ponto_inicial
        self.x, self.y, self.z, self.arestas = [], [], [], []

    def gera_cilindro(self):
        # Configurando os parâmetros do cilindro
        resolucao = (
            10  # Resolução do cilindro (quanto maior, mais suave será a superfície)
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

    def plota_cilindro(self, cor = 'b'):
        plotaSolido([self.x, self.y, self.z], self.arestas, cor)


# Cone #ROTACIONADO
radius = 1.0
height = 2 * radius
num_slices = 15
ponto_inicial_cone = [9, 6, 4]

cone = Cone(radius, height, num_slices, ponto_inicial_cone)
cone.gerar_cone()
cone.plota_cone()
cone.rotacionar_cone(90, 0, 0)
cone.plota_cone(cor = 'g')

# Cubo #ESCALADO e DESLOCADO
raio_cubo = 1
ponto_inicial_cubo = [-6, -8, -7]

cubo = Cubo(raio_cubo, ponto_inicial_cubo)
cubo.gera_cubo()
cubo.plota_cubo()
cubo.deslocar_cubo(1,0,1)
cubo.plota_cubo(cor = 'y')


# cubo.escalar_cubo(2, 2, 2)
# cubo.plota_cubo()

# Tronco Piramide
tronco = Tronco_piramide(2, 1, 3, (-9, -3, -8))
tronco.gera_tronco()
tronco.plota_tronco()
# posicao_final = (2, 3, 1)
tronco.rotacionar_tronco(45, 0, 0)
tronco.plota_tronco("g")
# tronco.plota_tronco()

# Esfera

# Cria um objeto Esfera com raio 1 e ponto inicial (0, 0, 0)
esfera = Esfera(1, (7, -2, -6))

# Plota a esfera original
esfera.gera_esfera()
esfera.plota_esfera()

# Aplica a escala de fator 2 em todos os eixos
esfera.escalar_esfera(2, 2, 2)
esfera.plota_esfera('r')

# Cilindro  #ESCALADO 
raio_cilindro = 1
altura_cilindro = 2
ponto_inicial_cilindro = [2, 4, 3]
cilindro = Cilindro(raio_cilindro, altura_cilindro, ponto_inicial_cilindro)
cilindro.gera_cilindro()
cilindro.plota_cilindro()


cilindro.escalar_cilindro(2, 2, 2,)  # esse valor 2 é o valor do "fator"
cilindro.plota_cilindro("r")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
