""""
Write a program to compute both the local and global clustering coefficients of a graph. Your
program should take two command line arguments: the name of a graph file (in our usual
modified, unweighted DIMACS format) followed by the index of a vertex. It should output the
local clustering coefficient of the vertex as well as the global clustering coefficient of the entire
graph. A user should see something similar to the following when running your program:

>./clustercoefficient graph.txt 5
Local coefficient: 0.2
Global coefficient: 0.42154
"""
import sys
import time

if len(sys.argv) != 3:
    print(
        "Bad input. Please execute the program as: \n./python3 clustercoefficient graphFile.txt\nWhere graphFile is a DIMACS file.")
    exit(1)


def convertLineToIntegers(line):
    [firstString, secondString] = line.replace('\n', '').split('\t')
    firstInt = int(firstString)
    secondInt = int(secondString)

    return (firstInt, secondInt)


def printMatrix(numVertices, matrix):
    for i in range(numVertices):
        for j in range(numVertices):
            print(matrix[i][j], end=' ')

        print()


def depthFirst(graph, node, visited):
    time.sleep(1)
    print(visited)
    print(f"{node + 1}", end=' ')
    for index in range(len(graph[node])):
        if graph[node][index] == 1 and index not in visited:
            visited = visited + [index]
            visited = depthFirst(graph, index, visited)

    return visited


def depthFirstSearchToDepthD(graph, node, visited, path, depth, sought, printPath):
    if depth == 2:
        # does visited consist of the values we want? #
        # for the cyclic graph - does 6 ad d2 have 6 and 7?

        print("Depth is 2")

        return visited, 0

    for index in range(len(graph[node])):
        if graph[node][index] == 1 and index not in visited:
            visited = visited + [index]
            path = path + [index + 1]
            (visited, flag) = depthFirstSearchToDepthD(
                graph, index, visited, path, depth - 1, sought, printPath)

            if (flag == 1):
                return (visited, flag)

            path.pop()
            visited.pop()
    return (visited, 0)


def idDFS(graph, node, sought, maxDepth=3):
    depth = 1
    while depth < maxDepth:
        printPath = False
        if depth > 4:
            printPath = True
        (visited, flag) = depthFirstSearchToDepthD(
            graph, node, [node], [node + 1], depth, sought, printPath)

        if flag == 1:
            return

        depth = depth + 1


def calculateDegrees(adjacencyMatrix):
    size = len(adjacencyMatrix)
    degrees = []

    for i in range(size):
        sumOfColumns = 0
        for j in range(size):
            if adjacencyMatrix[i][j] > 0:
                sumOfColumns = sumOfColumns + 1

        degrees.append(sumOfColumns)

    return degrees


def argsort(seq):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]


def getNeighbors(adjacencyMatrix, startingNode, degree):
    nodes = adjacencyMatrix[startingNode]
    returnList = [-1] * degree
    index = 0

    for i in range(len(nodes)):
        if nodes[i] > 0:
            returnList[index] = i
            index = index + 1

    return returnList


def colorNode(adjacencyMatrix, degree, nodeIndex, colorationMatrix):
    physicalNeighbors = getNeighbors(adjacencyMatrix, nodeIndex, degree)
    colorList = [0] * len(physicalNeighbors)
    color = 1

    for index in range(len(physicalNeighbors)):
        colorList[index] = colorationMatrix[physicalNeighbors[index]]

    while True:
        if color in colorList:
            color = color + 1
        else:
            break

    colorationMatrix[nodeIndex] = color
    # print(nodeIndex+1, color)
    return colorationMatrix


def colorizeGraph(adjacencyMatrix, degreeList, colorizationList=None):
    # calculate degree value
    coloredGraph = [-1] * len(adjacencyMatrix)

    if colorizationList is None:
        raise ValueError("Bad Colorization List")

    for index in colorizationList:
        coloredGraph = colorNode(
            adjacencyMatrix, degreeList[index], index, coloredGraph)

    return (coloredGraph)


def calculateVertexCover(adjacencyMatrix, vertexPath, degreeList):
    verticesInPath = []
    vertexCover = 0
    coloredGraph = colorizeGraph(adjacencyMatrix, degreeList, vertexPath)

    for index in range(len(coloredGraph)):
        if coloredGraph[index] == 1:
            vertexCover = vertexCover + 1
            verticesInPath.append(index)

    return (vertexCover, verticesInPath)


def calculateVertexCoverForAllPermutations(adjacencyMatrix, vertexPermutations):
    smallestCover = 100
    vertices = None
    degreeList = calculateDegrees(adjacencyMatrix)

    for permutation in vertexPermutations:
        (vertexCover, verticesInPath) = calculateVertexCover(
            adjacencyMatrix, permutation, degreeList)
        if vertexCover < smallestCover:
            smallestCover = vertexCover
            vertices = verticesInPath

    print(f"Smallest set is: ", smallestCover)
    print(f"Vertices are: ", vertices)


def calcVertexCover(edgeList, permutation):
    vertexCoverings = []
    edgeSet = set(edgeList)

    for vertex in permutation:
        for edge in list(edgeSet):
            if f'{vertex}' in edge:
                vertexCoverings.append(vertex)
                edgeSet.discard(edge)

            if len(edgeSet) == 0:
                return set(vertexCoverings)

    return "INTERESTING"


def removeHighestDegreeVertex(originalEdgeList, vertexCover, depth):
    (adjacencyMatrix, edgeList) = createAdjacencyMatrix(originalEdgeList)
    degreeList = calculateDegrees(adjacencyMatrix)

    zippedList = zip(degreeList, range(len(adjacencyMatrix)))
    sortedZippedList = sorted(zippedList)
    shortestDegreeToLargest = [element for _, element in sortedZippedList]
    largest = shortestDegreeToLargest[-1]

    edgeSet = set(edgeList)
    vertexCover.append(largest)
    for edge in list(edgeSet):
        if f"{largest}" in edge:
            edgeSet.discard(edge)

    if len(edgeSet) == 0 or depth == 10:
        print(vertexCover, depth)
        return

    removeHighestDegreeVertex(list(edgeSet), vertexCover, depth + 1)


def calculateNonPermutationVertexCover(originalEdgeList):
    # calculate degrees
    # drop all degrees of value 1

    (adjacencyMatrix, edgeList) = createAdjacencyMatrix(originalEdgeList)
    edgeSet = set(edgeList)
    degreesList = calculateDegrees(adjacencyMatrix)

    for vertex in range(len(degreesList)):
        if degreesList[vertex] == 1:
            for edge in list(edgeSet):
                if f"{vertex}" in edge:
                    edgeSet.discard(edge)

    removeHighestDegreeVertex(list(edgeSet), [], 1)


def convertFromFileToEdgeList(fileName):
    dimacs = open(fileName, 'r')

    firstLine = dimacs.readline()
    (numVertices, numEdges) = convertLineToIntegers(firstLine)

    edgeList = []

    for line in dimacs.readlines():
        (vertexOne, vertexTwo) = convertLineToIntegers(line)
        idxOne = vertexOne - 1
        idxTwo = vertexTwo - 1

        edgeList.append(f"{idxOne},{idxTwo}")

    dimacs.close()

    return (edgeList, numVertices)


def createAdjacencyMatrix(edgeList):

    adjacencyMatrix = []
    newEdgeList = []

    for i in range(numVertices):
        adjacencyMatrix.append([0] * numVertices)

    for line in edgeList:
        (vertexOne, vertexTwo) = line.split(',')
        idxOne = int(vertexOne)
        idxTwo = int(vertexTwo)

        newEdgeList.append(f"{idxOne},{idxTwo}")
        adjacencyMatrix[idxOne][idxTwo] = 1
        adjacencyMatrix[idxTwo][idxOne] = 1

    return (adjacencyMatrix, newEdgeList)


def calculateVertexCover_properExhausitve(edgesList, permutationVertices):
    smallestCover = 100
    covering = None

    for permutation in permutationVertices:
        vertexCovering = calcVertexCover(edgeList, permutation)
        if len(vertexCovering) < smallestCover:
            smallestCover = len(vertexCovering)
            covering = vertexCovering

    print("Smallest Covering: ", smallestCover)
    print("Covering: ", covering)



def calculateClusterCoefficients(originalEdgeList, localVertex):
    (adjacencyMatrix, edgeList) = createAdjacencyMatrix(originalEdgeList)
    degreesList = calculateDegrees(adjacencyMatrix)


    openTriplets = set()
    closedTriplets = set()
    connectedNeighbors = 0

    for vertex in range(len(adjacencyMatrix)):
        neighbors = getNeighbors(adjacencyMatrix, vertex, degreesList[vertex])

        for neighbor in neighbors:
            neighborsNeighbor = getNeighbors(adjacencyMatrix, neighbor, degreesList[neighbor])

            for secondaryNeighbor in neighborsNeighbor:
                if secondaryNeighbor == neighbor or secondaryNeighbor == vertex:
                    continue

                if vertex == localVertex-1:
                    if secondaryNeighbor in neighbors:
                        connectedNeighbors = connectedNeighbors + 1
                if secondaryNeighbor in neighbors and f"{vertex+1} - {secondaryNeighbor + 1} - {neighbor + 1}" not in closedTriplets:
                    closedTriplets.add(f"{vertex+1} - {neighbor+1} - {secondaryNeighbor+1}")

                elif secondaryNeighbor not in neighbors and f"{vertex+1} - {secondaryNeighbor+1} - {neighbor+1}" not in openTriplets and \
                        f"{neighbor+1} - {vertex+1} - {secondaryNeighbor+1}" not in openTriplets and \
                        f"{neighbor+1} - {secondaryNeighbor+1} - {vertex+1}" not in openTriplets and \
                        f"{secondaryNeighbor+1} - {neighbor+1} - {vertex+1}" not in openTriplets and \
                        f"{secondaryNeighbor+1} - {vertex+1} - {neighbor+1}" not in openTriplets:

                    openTriplets.add(f"{vertex+1} - {neighbor+1} - {secondaryNeighbor+1}")

    globalClusteringCoeff = len(closedTriplets) / ( len(openTriplets) + len(closedTriplets) )

    denom=   ( degreesList[localVertex-1] * (degreesList[localVertex-1] - 1 ))
    num = 2 * (connectedNeighbors / 2)

    localClusteringCoeff = 0.0 if num == 0 and denom == 0 else num / denom

    print("Global Clustering Coeff", globalClusteringCoeff)
    print("Local Clustering Coeff", localClusteringCoeff)


    # print("Open Triplets: ", openTriplets)
    # print("Closed Triplets: ", closedTriplets)




fileName = sys.argv[1]
localVertex = sys.argv[2]

adjacencyMatrix = []
(edgeList, numVertices) = convertFromFileToEdgeList(fileName)

calculateClusterCoefficients(edgeList, int(localVertex))
