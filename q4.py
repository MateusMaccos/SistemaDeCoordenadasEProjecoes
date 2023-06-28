import numpy as np
import matplotlib.pyplot as plt
import q3


def plotaSolido2D(ax, pontos, arestas, color="g"):
    for aresta in arestas:
        x = [pontos[0][aresta[0]], pontos[0][aresta[1]]]
        y = [pontos[1][aresta[0]], pontos[1][aresta[1]]]
        ax.plot(x, y, color)


cubeVertexListCamera = q3.cubeVertexListCamera.T
cilindroVertexListCamera = q3.cilindroVertexListCamera.T
coneVertexListCamera = q3.coneVertexListCamera.T
esferaVertexListCamera = q3.esferaVertexListCamera.T
troncoVertexListCamera = q3.troncoVertexListCamera.T

near = 0.1  # Valor de near
far = 100.0  # Valor de far


def TransformacaoEmPerspectiva(vertices, alpha, near, far):
    matrizHomogenea = np.vstack((vertices, np.ones((1, vertices.shape[1]))))
    verticesEmPerspectiva = np.zeros_like(matrizHomogenea)

    for i in range(matrizHomogenea.shape[1]):
        verticesEmPerspectiva[:, i] = (
            np.array(
                [
                    [1 / (matrizHomogenea[2, i] * np.tan(alpha / 2)), 0, 0, 0],
                    [0, 1 / (matrizHomogenea[2, i] * np.tan(alpha / 2)), 0, 0],
                    [
                        0,
                        0,
                        (near + far) / (near - far),
                        (2 * near * far) / (near - far),
                    ],
                    [0, 0, -1, 0],
                ]
            )
            @ matrizHomogenea[:, i]
        )

    verticesEmPerspectiva = verticesEmPerspectiva[:-1, :]
    verticesEmPerspectiva[-1, :] = 0

    return verticesEmPerspectiva


# Exemplo de matriz de vértices 3D
vertices_cubo = np.array(cubeVertexListCamera)
vertices_cone = np.array(coneVertexListCamera)
vertices_cilindro = np.array(cilindroVertexListCamera)
vertices_tronco = np.array(troncoVertexListCamera)
vertices_esfera = np.array(esferaVertexListCamera)

# Ângulo alpha em radianos
alpha = np.pi / 2

# Chamada da função de transformação em perspectiva
vertices_transformados_cubo = TransformacaoEmPerspectiva(
    vertices_cubo, alpha, near, far
)
# Chamada da função de transformação em perspectiva
vertices_transformados_cone = TransformacaoEmPerspectiva(
    vertices_cone, alpha, near, far
)
# Chamada da função de transformação em perspectiva
vertices_transformados_cilindro = TransformacaoEmPerspectiva(
    vertices_cilindro, alpha, near, far
)
# Chamada da função de transformação em perspectiva
vertices_transformados_tronco = TransformacaoEmPerspectiva(
    vertices_tronco, alpha, near, far
)
# Chamada da função de transformação em perspectiva
vertices_transformados_esfera = TransformacaoEmPerspectiva(
    vertices_esfera, alpha, near, far
)

# Criação do plot 2D
fig = plt.figure()
ax = fig.add_subplot(111)

# Criação do plot 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# Plota cubo original
# q3.plotaSolido(ax, cubeVertexListCamera, q3.cubo.arestas)

# Plota cubo projetado
plotaSolido2D(ax, vertices_transformados_cubo, q3.cubo.arestas)
# q3.plotaSolido(ax, vertices_transformados_cubo, q3.cubo.arestas)

# Plota cilindro original
# q3.plotaSolido(ax, cilindroVertexListCamera, q3.cilindro.arestas, "b")

# Plota cilindro projetado
plotaSolido2D(ax, vertices_transformados_cilindro, q3.cilindro.arestas, "b")
# q3.plotaSolido(ax, vertices_transformados_cilindro, q3.cilindro.arestas, "b")

# Plota tronco original
# q3.plotaSolido(ax, troncoVertexListCamera, q3.tronco.arestas, "c")

# Plota tronco projetado
plotaSolido2D(ax, vertices_transformados_tronco, q3.tronco.arestas, "c")
# q3.plotaSolido(ax, vertices_transformados_tronco, q3.tronco.arestas, "c")

# Plota esfera original
# q3.plotaSolido(ax, esferaVertexListCamera, q3.esfera.arestas, "m")

# Plota esfera projetado
plotaSolido2D(ax, vertices_transformados_esfera, q3.esfera.arestas, "m")
# q3.plotaSolido(ax, vertices_transformados_esfera, q3.esfera.arestas, "m")

# Plota tronco original
# q3.plotaSolido(ax, coneVertexListCamera, q3.cone.arestas, "k")

# Plota tronco projetado
plotaSolido2D(ax, vertices_transformados_cone, q3.cone.arestas, "k")
# q3.plotaSolido(ax, vertices_transformados_cone, q3.cone.arestas, "k")

camera = q3.eye
# Plot da camera no 3d
# ax.scatter(camera[0], camera[1], camera[2], c="r", label="eye")

at = q3.soma
# Plot do at no 3d
# ax.scatter(at[0], at[1], at[2], c="y", label="at")

# Configurações do plot
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()

# Exibição do plot
plt.show()
