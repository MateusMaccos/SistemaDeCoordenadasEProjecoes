import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# def rotacionar(angle_x, angle_y, angle_z, x, y, z):
#     # Rotaciona os pontos (x, y, z) em torno dos eixos X, Y e Z.

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
#     # Escala os pontos (x, y, z) de acordo com os fatores de escala sx, sy e sz.

#     matriz_escala = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
#     pontos_scaled = matriz_escala.dot([x, y, z])

#     pontos_scaled[0] += ponto_inicial[0] * (1 - sx)
#     pontos_scaled[1] += ponto_inicial[1] * (1 - sy)
#     pontos_scaled[2] += ponto_inicial[2] * (1 - sz)
#     return pontos_scaled[0], pontos_scaled[1], pontos_scaled[2]


# def deslocar(dx, dy, dz, x, y, z):
#     # Desloca os pontos (x, y, z) nas direções dx, dy e dz.

#     T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])
#     complete_row = np.full((1, len(x)), 1)[0]
#     new_matrix = T.dot([x, y, z, complete_row])
#     x = new_matrix[0]
#     y = new_matrix[1]
#     z = new_matrix[2]
#     return x, y, z


def plotaSolido(pontos, arestas):
    # Plota o sólido definido pelos pontos e arestas no gráfico 3D.

    for aresta in arestas:
        x = [pontos[0][aresta[0]], pontos[0][aresta[1]]]
        y = [pontos[1][aresta[0]], pontos[1][aresta[1]]]
        z = [pontos[2][aresta[0]], pontos[2][aresta[1]]]
        ax.plot(x, y, z, "b")


class Cone:
    def __init__(self, radius, height, num_slices, ponto_inicial):
        """
        Inicializa um objeto Cone com os parâmetros fornecidos.

        Args:
            radius (float): Raio do cone.
            height (float): Altura do cone.
            num_slices (int): Número de fatias (seções) do cone.
            ponto_inicial (list): Lista contendo as coordenadas iniciais do cone.

        Returns:
            None
        """
        self.radius = radius
        self.height = height
        self.num_slices = num_slices
        self.ponto_inicial = ponto_inicial
        self.pontosX, self.pontosY, self.pontosZ, self.arestas = [], [], [], []

    def generate_cone(self):
        # Gera as coordenadas (x, y, z) do cone.

        theta = np.linspace(0, 2 * np.pi, self.num_slices)
        z = np.linspace(0, self.height, self.num_slices)

        x = np.empty((self.num_slices, self.num_slices))
        y = np.empty((self.num_slices, self.num_slices))
        z_2d = np.empty((self.num_slices, self.num_slices))

        # diminuir o tamanho da base do cone ao longo da altura
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
        # Gera os pontos e arestas do cone.

        contador = 0
        x, y, z = self.generate_cone()
        print(x)
        for i in range(self.num_slices):
            for j in range(len(x[i])):
                self.pontosX.append(x[i][j])
                self.pontosY.append(y[i][j])
                self.pontosZ.append(z[i][j])
                # H
                if j < (len(x[i]) - 1):
                    self.arestas.append([contador, contador + 1])
                contador += 1
            # V
            for circles in range(len(x[i])):
                self.arestas.append([i, i + self.num_slices * circles])

    # def rotacionar_cone(self, angle_x, angle_y, angle_z):
    #     # Rotaciona o cone em torno dos eixos X, Y e Z.

    #     self.pontosX, self.pontosY, self.pontosZ = rotacionar(
    #         angle_x, angle_y, angle_z, self.pontosX, self.pontosY, self.pontosZ
    #     )

    # def deslocar_cone(self, dx, dy, dz):
    #     # Desloca o cone nas direções dx, dy e dz.

    #     self.pontosX, self.pontosY, self.pontosZ = deslocar(
    #         dx, dy, dz, self.pontosX, self.pontosY, self.pontosZ
    #     )

    # def escalar_cone(self, sx, sy, sz):
    #     # Escala o cone de acordo com os fatores de escala sx, sy e sz.

    #     self.pontosX, self.pontosY, self.pontosZ = escalar(
    #         sx, sy, sz, self.pontosX, self.pontosY, self.pontosZ, self.ponto_inicial
    #     )

    def plota_cone(self):
        # Plota o cone no gráfico 3D.

        plotaSolido([self.pontosX, self.pontosY, self.pontosZ], self.arestas)


# Cone
radius = 1.0
height = 2 * radius
num_slices = 10
ponto_inicial_cone = [0, 0, 0]

cone = Cone(radius, height, num_slices, ponto_inicial_cone)
cone.gerar_cone()
cone.plota_cone()

plt.show()
