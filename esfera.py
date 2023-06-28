import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


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


def plotaSolido(pontos, arestas):
    for aresta in arestas:
        x = [pontos[0][aresta[0]], pontos[0][aresta[1]]]
        y = [pontos[1][aresta[0]], pontos[1][aresta[1]]]
        z = [pontos[2][aresta[0]], pontos[2][aresta[1]]]
        ax.plot(x, y, z, "b")


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

        # Percorre Horizontalmente ( L -> O )
        for _ in range(num_pontos + 1):
            phi = 0.0
            # Percorre Verticalmente ( N -> S )
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

        # Traça as arestas da esfera
        for i in range(num_pontos):
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

    def plota_esfera(self):
        plotaSolido([self.x, self.y, self.z], self.arestas)


# Esfera

# Cria um objeto Esfera com raio 1 e ponto inicial (0, 0, 0)
esfera = Esfera(1, (7, -2, -6))

# Plota a esfera original
esfera.gera_esfera()
# esfera.escalar_esfera(2, 2, 2)
# esfera.deslocar_esfera( 1, 1, 1)
# esfera.rotacionar_esfera(90, 90, 90)
esfera.plota_esfera()

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
