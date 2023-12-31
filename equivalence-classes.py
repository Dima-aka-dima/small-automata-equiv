import numpy as np
import itertools
import functools
import networkx as nx

## useful functions
compose = lambda f, g: lambda x: f(g(x))
composeList = lambda fs: functools.reduce(compose, fs)
mapList = lambda f, xs: list(map(f, xs))
mapFunctions = lambda fs, x: mapList(lambda f, x=x: f(state), fs)


## Game of Life logic

def updateCell(cell, neighbourCount):
    if cell == 0:
        if neighbourCount == 3: return 1
        return 0
    if (neighbourCount == 2) or (neighbourCount == 3): return 1
    return 0
    
def update(state):
    newState = np.zeros(state.shape, dtype = int)
    height, width = state.shape
    
    for i in range(height):
        for j in range(width):
            neighbourCount = 0
            
            for deltaI in [-1, 0, 1]:
                for deltaJ in [-1, 0, 1]:
                    if (deltaI == 0) and (deltaJ == 0): continue
                    neighbourCount += state[(i + deltaI) % height][(j + deltaJ) % width]
            newState[i][j] = updateCell(state[i][j], neighbourCount)
    return newState


D4Transformations = [
    lambda s: s,
    lambda s: np.rot90(s, k = 1),
    lambda s: np.rot90(s, k = 2),
    lambda s: np.rot90(s, k = 3),
    lambda s: np.flip(s, axis = 0),
    lambda s: np.flip(s, axis = 1),
    lambda s: np.flip(np.rot90(s, k = 1), axis = 0),
    lambda s: np.flip(np.rot90(s, k = 1), axis = 1)
]

oneTranslations = [
    lambda s: np.roll(s, shift = 1, axis = 0),
    lambda s: np.roll(s, shift = 1, axis = 1)
]

def getAllTransformations(shape):
    if shape == (1, 1): return [[lambda s: s]]
    if shape == (2, 2): return [[t] for t in D4Transformations]
    
    allTransformations = [] 
    
    for transform in D4Transformations:
        for nDown, nRight in itertools.product(range(shape[0]), range(shape[1])):
            allTransformations += [[transform] + [oneTranslations[0]]*nRight + [oneTranslations[1]]*nDown]
                
    return allTransformations

def getMatrixForTransformation(transformation, shape):
    matrix = np.zeros((shape[0]*shape[1], shape[0]*shape[1]), dtype = int)

    for i, x in enumerate(np.identity(shape[0]*shape[1], dtype = int)):
        matrix[i] = transformation(x.reshape(shape)).flatten()  
    return matrix.T



n = 4
shape = (n, n)


## make matrices that act on a flattened state
## they are a representation of our group
transformationsAsSequences = getAllTransformations(shape)
transformations = mapList(composeList, transformationsAsSequences)

matrices = mapList(lambda t: getMatrixForTransformation(t, shape), transformations)
matrices = np.array(matrices)


## count equivalence classes
currentClass = 0
classes = -1 + np.zeros(2**(n**2), dtype = int)
minimalStates = []
powersOfTwo = 2**np.arange(n**2)[::-1]

for index in range(2**(n**2)):
    if classes[index] != -1: continue
    
#     print(f"{index:8d}", end = '\r')
    
    state = ((index & powersOfTwo) > 0).astype(int)
    minimalStates += [state]
    
    nextStates = np.einsum('nij,j->ni', matrices, state)
    nextIndices = np.unique(np.einsum('ni,i->n', nextStates, powersOfTwo))

    classes[nextIndices] = currentClass

    currentClass += 1
    
nClasses = currentClass

print(nClasses)

## create a graph
## now everything you want to know is contained in a graph (and `minimalStates`)
edges = np.zeros(nClasses, dtype = int)
for index, state in enumerate(minimalStates):
    nextState = update(state.reshape(shape)).flatten()
    nextStateAsIndex = nextState.dot(powersOfTwo)
    
    nextClass = classes[nextStateAsIndex]
    
    edges[index] = nextClass
    
edges = np.vstack([np.arange(edges.shape[0]), edges]).T

graph = nx.DiGraph()
graph.add_nodes_from(np.arange(nClasses, dtype = int))
graph.add_edges_from(edges)


cycles = list(nx.simple_cycles(graph))
print(cycles)

## count the number of classes
longestOrder = 0
nOrbits = 0
for g in transformations:
    
    state = np.arange(shape[0]*shape[1]).reshape(shape)
    originalState = state.copy()
    states = [originalState]
    while True:
        state = g(state)
        if (state == originalState).all(): break
        states += [state]
        
    states = np.array(states)
    degrees = np.zeros(shape, dtype = np.uint64)
    for i, j in itertools.product(range(shape[0]), range(shape[1])):

        atPlace = np.where(states[:,i,j] == states[0,i,j])[0]
        degrees[i][j] = states.shape[0]
        if atPlace.shape[0] > 1: degrees[i][j] = atPlace[1] - atPlace[0]

    nFixedByElement = np.prod(2**(1/degrees)).astype(np.uint64)
#     print(nFixedByElement, states.shape[0])
    nOrbits += nFixedByElement
print(nOrbits/(8*shape[0]*shape[1]))
