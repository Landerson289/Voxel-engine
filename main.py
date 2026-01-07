### Optimisations
# Reducing unnecessary calculations
# Pre calculate important values

### Refactors
# Make functions for common things

### Improvements
# Reflection and refraction in one pixel (double beams). Do this without repeating code
# Adjustable voxel sizes
# Simulate absorbtion more accurately
# World creation and distruction

### Block improvements
# Water reflects or transmits depending on critical angle to surface
# Clouds use Acerola smoke grenade tactics (or approximation)
# Leaves are transparent and light reactive.
# Grass and wood are sharper
# Include noise in water and cloud textures

### Bugs
# Lines in the water

import pygame
import time
import random
import math as maths
import octree
from functools import partial

count = 0




def normalise(vector):
    if vector == [0,0,0]: return vector
    s = 0
    for i in vector:
        s += i**2
    s = maths.sqrt(s)
    return [i/s for i in vector]

'''
random_normalised_vectors = [normalise([random.uniform(-1,1) for i in range(3)]) for i in range(10**7)]
random_normalised_vectors_index = 0
number_of_random_normalised_vectors = len(random_normalised_vectors)

#do random normal vectors to each plane in the same way

random_normal_vectors = 
'''
import numpy as np

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

perlin_grid = generate_perlin_noise_2d([128,128], [32,32])

random0to1 = [random.uniform(0, 1) for i in range(10**7)]
randomNeg1to1 = [random.uniform(-1,1) for i in range(10**7)]

random0to1_index = 0
randomNeg1to1_index = 0

def get_random_normalised_vector():
    global randomNeg1to1_index
    #return [random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]
    #random_normalised_vectors[random_normalised_vectors_index%number_of_random_normalised_vectors]
    #random_normalised_vectors_index += 1
    randomNeg1to1_index += 3
    randomNeg1to1_index %= len(randomNeg1to1)
    return [randomNeg1to1[randomNeg1to1_index-2],randomNeg1to1[randomNeg1to1_index-1],randomNeg1to1[randomNeg1to1_index]]

def get_random_normalised_vector_old():
    return [random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]

def reflect_in_face(point, direction): # Reflects the direction along any voxel faces the point may be sitting on
    newDirection = []
    for i in range(3):
        #if round(point[i]%1, 2) in [0,1]:
        if point[i]%1 < 0.005 or point[i]%1 > 0.995:
            newDirection.append(-direction[i])
        else:
            newDirection.append(direction[i])
    return newDirection

def diffuse_reflection(point):
    global randomNeg1to1_index, random0to1_index
    vector = []
    for i in range(3):
        if point[i]%1 > 0.995:
            #vector.append(random0to1[random0to1_index%len(random0to1)])
            #random0to1_index += 1
            vector.append(random.uniform(0,1))
        elif point[i]%1 < 0.005:
            #vector.append(-random0to1[random0to1_index%len(random0to1)])
            #random0to1_index += 1
            vector.append(random.uniform(-1,0))
        else:
            #vector.append(random0to1[randomNeg1to1_index%len(randomNeg1to1)])
            #randomNeg1to1_index += 1
            vector.append(random.uniform(-1,1))
    return vector

def blend(colour1, colour2, coefficient):
    return [coefficient*colour1[i] + (1-coefficient)*colour2[i] for i in range(3)]

def brightnessAdjustment(colour1, colour2): # Adjust colour 2 so that it is the same brightness as colour 1
    # This is a first attempt to mimic light bouncing off of an object and retaining its characterists
    #brightness1 = (colour1[0]+colour1[1]+colour1[2])/3
    #brightness2 = (colour2[0]+colour2[1]+colour2[2])/3
    #resultantBrightness = 0.15*(brightness1+brightness2)/(brightness2)

    sum2 = colour2[0]+colour2[1]+colour2[2]
    resultantBrightness = 0.25*(colour1[0]+colour1[1]+colour1[2] + sum2)/sum2
    return [max(0,min(255, resultantBrightness * colour2[i])) for i in range(3)]


textures = pygame.image.load("3D projects/pixil-layer-Background - Frame1.png")


textureCoords = {
    "dirt" : (2,0),
    "grassTop" : (0,0),
    "grassSide" : (1,0),
    "stone" : (3,0),
    "bedrock" : (0,1),
    "woodSide" : (1,1),
    "woodTop" : (2,1),
    "leaf" : (3,1),
    "gold" : (0,2),
    "emerald" : (1,2),
    "diamond" : (2,2),
    "sand" : (3,2),
    "brick" : (0,3),
    "cobblestone" : (1,3),
    "planks" : (2,3),
    "pride" : (3,3),
}


def get_pixel_from_texture(blockType, point):
    #if round(point[2]%1, 2) in [0,1]:
    if point[2]%1 < 0.005 or point[2]%1 > 0.995:
        X,Y = textureCoords[blockType if blockType not in ["grass", "wood"] else blockType+"Side"]
        x,y = (int(8*X+(point[0]%1)*8),int(8*Y+(point[1]%1)*8))
    #elif round(point[1]%1, 2) in [0,1]:
    elif point[1]%1 < 0.005 or point[1]%1 > 0.995:
        X,Y = textureCoords[blockType if blockType not in ["grass", "wood"] else blockType+"Top"]
        x,y = (int(8*X+(point[0]%1)*8),int(8*Y+(point[2]%1)*8))
    else:
        X,Y = textureCoords[blockType if blockType not in ["grass", "wood"] else blockType+"Side"]
        x,y = (int(8*X+(point[2]%1)*8),int(8*Y+(point[1]%1)*8))
    colour = textures.get_at((x,y))
    return colour

direction_functions = {
    "no_effect" : lambda point, dir : dir,
    "specular_reflection" : lambda point, dir : reflect_in_face(point, dir),
    "diffuse_reflection" : lambda point, dir : diffuse_reflection(point),
    "halt" : lambda point, dir : [0,0,0],
    "mixed_optics" : lambda point, dir : random.choice([dir, reflect_in_face(point, dir)]),
}

sqrt3 = maths.sqrt(3)

colour_functions = {
    "no_effect" : lambda in_colour, point, dir, rayLength : in_colour,
    "colour" : lambda colour, in_colour, point, dir, rayLength : colour,
    "blend" : lambda colour, coefficient, in_colour, point, dir, rayLength : blend(colour, in_colour, coefficient),
    "random" : lambda in_colour, point, dir, rayLength : (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
    "fog_blend": lambda colour, in_colour, point, dir, rayLength : blend(colour, in_colour, rayLength/sqrt3),
    "texture": lambda texture, in_colour, point, dir, rayLength : get_pixel_from_texture(texture, point),
    "blend_texture" : lambda texture, coefficient, in_colour, point, dir, rayLength : blend(get_pixel_from_texture(texture, point), in_colour, coefficient),
    "light_responsive_colour" : lambda colour, in_colour, point, dir, rayLength : brightnessAdjustment(in_colour, colour),
    "light_responsive_texture" : lambda texture, in_colour, point, dir, rayLength : brightnessAdjustment(in_colour, get_pixel_from_texture(texture, point)),
    "perlin_noise" : lambda in_colour, point, dir, rayLength : [in_colour[i]*0.75+in_colour[i]*(0.25*perlin_grid[int((point[0]*16)%128)][int((point[2]*16)%128)]) for i in range(2)] + [in_colour[2]],
}

blocks = { # Name : [Direction_effect, Colour_effect, direction_arguments, colour_arguments
    "air" : ["no_effect", "no_effect", [], []],#[(150,150,255), 0.01]], # Blends the background into the sky
    "mirror" : ["specular_reflection", "blend", [], [(255,255,255), 0.1]], # Mirror
    "noise" : ["halt", "random", [], []], # Returns a block of noise
    "mist" : ["no_effect", "fog_blend", [], [(255,255,255)]],
    "whiteLight" : ["halt", "colour", [], [(255,255,255)]], # White box of light
    #"colouredFaces", # Different faces have different colours
    "grass" : ["halt", "texture", [], ["grass"]], # Textured block
    "dirt" : ["halt", "texture", [], ["dirt"]], # Textured block
    "wood" : ["halt", "texture", [], ["wood"]], # Textured block
    "translucentLeaves" : ["no_effect", "blend_texture", [], ["leaf", 0.7]], # Translucent texture block
    "water" : ["specular_reflection", "perlin_noise", [], []],#[(0,0,255), 0.1]],
    "lightResponsiveRed" : ["diffuse_reflection", "light_responsive_colour", [], [(255,0,0)]],
    "lightResponsiveGrass" : ["diffuse_reflection", "light_responsive_texture", [], ["grass"]],
    "lightResponsiveWood" : ["diffuse_reflection", "light_responsive_texture", [], ["wood"]],
    "lightResponsiveDirt" : ["diffuse_reflection", "light_responsive_texture", [], ["dirt"]],
    #"Double reflector"
}

# create way to apply layers on top of each other to remove need for blend_texture etc.


applied_blocks = {}
for blockKey in blocks:
    block = blocks[blockKey]
    applied_blocks[blockKey] = [partial(direction_functions[block[0]], *block[2]), partial(colour_functions[block[1]], *block[3])]

N = 10000
blockKeys = []
for i in range(N):
    blockKeys.append(random.choice(list(blocks.keys())))

st = time.perf_counter()
for i in range(N):
    block = applied_blocks[blockKeys[i]]
    colour = block[1]((i,i,i), [0,0,0], [1,0,0], 1)
et = time.perf_counter()
print((et-st)/N)

def sky(direction):
    if (direction[1]+0.25)**2 + (direction[0]-0.25)**2 < 0.0025: # 0.5**2
        return (255,255,200)
    else:
        return (150,150,255)#(0,0,0)

sins = []
for i in range(1000):
    sins.append(maths.sin(i/10))

class World:
    def __init__(self):
        self.static = True
        s = time.perf_counter()
        self.generate_world()
        e = time.perf_counter()
        print("create array", e-s)

        s = time.perf_counter()
        self.octree = octree.Octree(self.voxel_array, 0, 0, 0, len(self.voxel_array))
        e = time.perf_counter()
        print("create octree", e-s)

        s = time.perf_counter()
        self.dynamic_blocks = []
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if self.is_block_dynamic((i,j,k)):
                        self.dynamic_blocks.append((i,j,k))
        
        del self.voxel_array
    
    def generate_world(self):
        WORLD_SIZE = 128
        self.voxel_array = []
        for i in range(WORLD_SIZE):
            self.voxel_array.append([])
            for j in range(WORLD_SIZE):
                self.voxel_array[-1].append([])
                for k in range(WORLD_SIZE):
                    
                    #colour = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    #condition = j > 32

                    #colour = (0,random.randint(150,255),0) if j > 5 else (random.randint(150,255),random.randint(150,255),random.randint(150,255))
                    #condition = j > 10+5*maths.sin(i/10)+5*maths.sin(k/10)
                    condition = j > 10 + 5*sins[i] + 5*sins[k]

                    #colour = random.choice(list(textureCoords.keys()))#"grass"
                    if j < 20:
                        #colour = "grass"
                        block = "lightResponsiveGrass"
                    elif j < 50:
                        block = "lightResponsiveDirt"
                    else:
                        block = "lightResponsiveStone"

                    #colour = "grass"

                    if condition == 0 and j > 13:
                        condition = 1
                        block = "water"

                    self.voxel_array[-1][-1].append(block if condition else "air")
                    #self.voxel_array[-1][-1].append((2*i, 2*j, 2*k) if j > 5+3*maths.sin(i+k) else 0)
                    #self.voxel_array[-1][-1].append((random.randint(0,255), random.randint(0,255), random.randint(0,255)) if j > 5 else 0)

                    #condition = (i == 12)
                    '''
                    condition = (11 < i < 15 and j == 11 and 8 < k < 12) or (11 < i < 15 and 7 < j < 11 and k == 12) or (i == 15 and 7 < j < 11 and 8 < k < 12)

                    if condition:#condition:
                        self.voxel_array[-1][-1].append("lightResponsiveRed")
                    else:
                        self.voxel_array[-1][-1].append("air")
                    '''
        print("Terrain Done")
        
        for i in range(WORLD_SIZE*WORLD_SIZE//128):
            x = random.randrange(0,WORLD_SIZE)
            z = random.randrange(14,WORLD_SIZE)
            voxel = "air"
            y = -1
            while voxel == "air":
                y += 1
                voxel = self.voxel_array[x][y][z]
                
                
            
            for i in range(5):
                y -= 1
                #self.set_voxel(x,y,z, "wood")
                self.voxel_array[int(x)][int(y)][int(z)] = "lightResponsiveWood"
                
            
            y -= 1
            for dy in range(3):
                for dx in range(-1,2):
                    for dz in range(-1,2):
                        if dx == 0 and dz == 0:
                            continue

                        #self.set_voxel(x+dx,y+dy,z+dz, "leaf")
                        try:
                            self.voxel_array[int(x+dx)][int(y+dy)][int(z+dz)] = "translucentLeaves"
                        except:
                            pass
        

        for i in range(WORLD_SIZE*WORLD_SIZE//8):
            v = 1
            while v != "air":
                x = random.randrange(0, WORLD_SIZE)
                y = random.randrange(0, 5)
                z = random.randrange(0, WORLD_SIZE)
                v = self.voxel_array[int(x)][int(y)][int(z)]
            self.voxel_array[int(x)][int(y)][int(z)] = "mist"

        #for x in range(128):
        #    for z in range(128):
        #        self.voxel_array[int(x)][0][int(z)] = "mirror"


        self.voxel_array[9][5][20] = "whiteLight"
        
    def is_block_dynamic(self, position):
        block = self.get_voxel(*position)[0]
        if block == "sand":
            if self.get_voxel(position[0], position[1]+1, position[2])[0] in ["air", "sand"]:
                return True
            else:
                return False
        return False

    def update_dynamics(self):
        new_dynamic_blocks = []
        for block_pos in self.dynamic_blocks:
            block_type, tree = self.get_voxel(*block_pos)

            if block_type == "mist":
                new_block_pos = block_pos[0], block_pos[1]+1, block_pos[2]
                self.set_voxel(*block_pos, "air")
                self.set_voxel(*new_block_pos, "mist")

                if self.is_block_dynamic(new_block_pos):
                    new_dynamic_blocks.append([*new_block_pos])
        self.dynamic_blocks = new_dynamic_blocks
            

    def get_voxel(self, pointX, pointY, pointZ):
        current_tree = self.octree
        voxel = current_tree.block
        size = current_tree.size
        half_size = size//2

        #print(pointX, pointY, pointZ)

        if not (0 <= pointX < size and 0 <= pointY < size and 0 < pointZ < size):
            return -1, None

        relativeX = pointX #- current_tree.x
        relativeY = pointY #- current_tree.y
        relativeZ = pointZ #- current_tree.z

        while voxel == None:
            if relativeZ < half_size:
                if relativeY < half_size:
                    if relativeX < half_size:
                        current_tree = current_tree.children[0]
                    else:
                        current_tree = current_tree.children[1]
                        relativeX -= half_size
                else:
                    if relativeX < half_size:
                        current_tree = current_tree.children[2]
                    else:
                        current_tree = current_tree.children[3]
                        relativeX -= half_size
                    relativeY -= half_size
            else:
                if relativeY < half_size:
                    if relativeX < half_size:
                        current_tree = current_tree.children[4]
                    else:
                        current_tree = current_tree.children[5]
                        relativeX -= half_size
                else:
                    if relativeX < half_size:
                        current_tree = current_tree.children[6]
                    else:
                        current_tree = current_tree.children[7]
                        relativeX -= half_size
                    relativeY -= half_size
                relativeZ -= half_size
            
            half_size //= 2
            voxel = current_tree.block
        
        return voxel, current_tree

    def set_voxel(self, pointX, pointY, pointZ, new_voxel):
        voxel, tree = self.get_voxel(pointX, pointY, pointZ)
        if voxel == new_voxel or voxel == -1:
            return
        self.static = False
        while tree.size != 1:
            tree.block = None
            tree.create_children()
            for child in tree.children:
                child.block = voxel
            #print([pointX, pointY, pointZ], [tree.x, tree.y, tree.z], tree.size)
            index = 4*(pointX-tree.z)//(tree.size/2) + 2*(pointY-tree.y)//(tree.size/2) + 1*(pointZ-tree.x)//(tree.size/2)
            #print(index)
            tree = tree.children[int(index)]
        
        tree.block = new_voxel

def cast_ray(start, rayDir, depth=1):
    global LAMBDA_COUNT, GET_COUNT
    #global count
    #count += 1

    if depth > 200:
        return (255,255,0)

    #GET_COUNT += 1
    voxelName, currentTree = world.get_voxel(*start)
    voxel = applied_blocks[voxelName]
    
    #LAMBDA_COUNT += 1
    newRayDir = voxel[0](start, rayDir)
    #newRayDir = normalise(newRayDir)

    if newRayDir == [0,0,0]:
        #LAMBDA_COUNT += 1
        return voxel[1](None, start, newRayDir, None)
    
    size = currentTree.size
    
    dl = 1000
    if newRayDir[0] > 0:
        temp = (size-start[0]%size)/newRayDir[0] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
    elif newRayDir[0] < 0:
        temp = -(start[0]%size)/newRayDir[0] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
    

    if newRayDir[1] > 0:
        temp = (size-start[1]%size)/newRayDir[1] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
    elif newRayDir[1] < 0:
        temp = -(start[1]%size)/newRayDir[1] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
        
    if newRayDir[2] > 0:
        temp = (size-start[2]%size)/newRayDir[2] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
    elif newRayDir[2] < 0:
        temp = -(start[2]%size)/newRayDir[2] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl

    end = [start[0], start[1], start[2]]
    DL = 0
    while int(start[0]) == int(end[0]) and int(start[1]) == int(end[1]) and int(start[2]) == int(end[2]):
        end[0] += dl*newRayDir[0]
        end[1] += dl*newRayDir[1]
        end[2] += dl*newRayDir[2]
        DL += dl
        dl = 0.01

    if not 0 <= end[0] < worldLength or not 0 <= end[1] < worldLength or not 0 <= end[2] < worldLength:
        return sky(newRayDir)
    
    in_colour = cast_ray(end, newRayDir, depth+1)
    #LAMBDA_COUNT += 1
    out_colour = voxel[1](in_colour, start, newRayDir, DL)
    return out_colour
    #return voxel[1](cast_ray(end, newRayDir, depth+1), start, newRayDir, DL)

def basic_cast(start, rayDir, depth=1):
    if depth > 200:
        return [-1,-1,-1]

    voxelName, currentTree = world.get_voxel(*start)
    voxel = applied_blocks[voxelName]
    
    if voxelName != "air":
        return start
    newRayDir = rayDir
    
    size = currentTree.size
    
    dl = 1000
    if newRayDir[0] > 0:
        temp = (size-start[0]%size)/newRayDir[0] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
    elif newRayDir[0] < 0:
        temp = -(start[0]%size)/newRayDir[0] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
    

    if newRayDir[1] > 0:
        temp = (size-start[1]%size)/newRayDir[1] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
    elif newRayDir[1] < 0:
        temp = -(start[1]%size)/newRayDir[1] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
        
    if newRayDir[2] > 0:
        temp = (size-start[2]%size)/newRayDir[2] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl
    elif newRayDir[2] < 0:
        temp = -(start[2]%size)/newRayDir[2] + 0.000001
        if dl > temp:
            dl = temp #if temp != 0 else dl

    end = [start[0], start[1], start[2]]
    DL = 0
    while int(start[0]) == int(end[0]) and int(start[1]) == int(end[1]) and int(start[2]) == int(end[2]):
        end[0] += dl*newRayDir[0]
        end[1] += dl*newRayDir[1]
        end[2] += dl*newRayDir[2]
        DL += dl
        dl = 0.01

    if not 0 <= end[0] < worldLength or not 0 <= end[1] < worldLength or not 0 <= end[2] < worldLength:
        return [-1,-1,-1]
    
    end = basic_cast(end, newRayDir, depth+1)
    return end

if __name__ == "__main__":
    st = time.perf_counter()
    world = World()

    WIDTH = 400
    HEIGHT = 400

    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    WIDTH = screen.get_width()
    HEIGHT = screen.get_height()

    RESOLUTION = 1

    effWIDTH = WIDTH//RESOLUTION
    effHEIGHT = HEIGHT//RESOLUTION

    worldLength = world.octree.size

    camPosX,camPosY,camPosZ = [64.5,5,64]
    #camDirX,camDirY,camDirZ = [0,0,1]

    yaw = maths.pi/4

    camDirX,camDirY,camDirZ = [maths.sin(yaw),0,maths.cos(yaw)]

    running = True

    holdingDown = False

    iteration = 0

    print("WORLD GENERATED", time.perf_counter() - st)

    buffer = []

    while running:
        st = time.perf_counter()
        
        LAMBDA_COUNT = 0
        GET_COUNT = 0
        for imageX in range(-effWIDTH//2, effWIDTH//2):
            for imageY in range(-effHEIGHT//2, effHEIGHT//2):
                pointX = camPosX-camDirX
                pointY = camPosY-camDirY
                pointZ = camPosZ-camDirZ

                rayDirX = camDirX + camDirZ * imageX / effWIDTH
                rayDirY = imageY/effHEIGHT + camDirY
                rayDirZ = camDirZ - camDirX * imageX / effWIDTH

                if pointX < 0 or pointY < 0 or pointZ < 0:
                    if pointX < 0:
                        dl = -pointX/rayDirX + 0.001
                    if pointY < 0:
                        temp = -pointY/rayDirY + 0.001
                        if dl > temp:
                            dl = temp
                    if pointZ < 0:
                        temp = -pointZ/rayDirZ + 0.001
                        if dl > temp:
                            dl = temp
                    pointX += dl * rayDirX
                    pointY += dl * rayDirY
                    pointZ += dl * rayDirZ

                if pointX < 0 or pointY < 0 or pointZ < 0:
                    colour = sky([rayDirX, rayDirY, rayDirZ])
                else:
                    colour = cast_ray([pointX, pointY, pointZ], [rayDirX, rayDirY, rayDirZ])


                #add check to see if dynamic objects and camera motion
                if world.static:
                    if iteration == 0:
                        current_colour = colour
                    else:
                        current_colour = screen.get_at((RESOLUTION*(imageX+effWIDTH//2), RESOLUTION*(imageY+effHEIGHT//2)))
                    
                    averageColour = [(iteration*current_colour[i] + colour[i])/(iteration+1) for i in range(3)]
                else:
                    averageColour = colour

                #pygame.draw.rect(screen, colour, (RESOLUTION*(imageX+effWIDTH//2), RESOLUTION*(imageY+effHEIGHT//2), RESOLUTION, RESOLUTION))
                screen.set_at((RESOLUTION*(imageX+effWIDTH//2), RESOLUTION*(imageY+effHEIGHT//2)), averageColour)

            pygame.draw.circle(screen, (50,50,50), (WIDTH//2, HEIGHT//2),5)
            pygame.display.update()
            

            for event in pygame.event.get(): # Can convert loop back to one main loop when framerate increases
                if event.type == pygame.MOUSEBUTTONDOWN:
                    holdingDown = time.perf_counter()
                elif event.type == pygame.MOUSEBUTTONUP:
                    buffer.append(event)
                    holdingDown = time.perf_counter() - holdingDown
                else:
                    buffer.append(event)
        world.static = True
        world.update_dynamics()

        et = time.perf_counter()
        print(et-st)

        vel = [0,0,0]

        for event in buffer:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    vel[2] = 1
                elif event.key == pygame.K_s:
                    vel[2] = -1
                elif event.key == pygame.K_a:
                    vel[0] = -1
                elif event.key == pygame.K_d:
                    vel[0] = 1
                elif event.key == pygame.K_UP:
                    vel[1] = -1
                elif event.key == pygame.K_DOWN:
                    vel[1] = 1
                elif event.key == pygame.K_q:
                    yaw -= maths.pi/12
                    camDirX,camDirY,camDirZ = [maths.sin(yaw),0,maths.cos(yaw)]
                elif event.key == pygame.K_e:
                    yaw += maths.pi/12
                    camDirX,camDirY,camDirZ = [maths.sin(yaw),0,maths.cos(yaw)]
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    running = False
                world.static = False

                camPosZ += (camDirZ * vel[2] - camDirX * vel[0])*0.5
                camPosY += vel[1]*0.5
                camPosX += (camDirZ * vel[0] + camDirX * vel[2])*0.5
            #elif event.type == pygame.MOUSEBUTTONDOWN:
            #    holdingDown = time.perf_counter()

            elif event.type == pygame.MOUSEBUTTONUP:
                rayDirX = camDirX + camDirZ * 0 / effWIDTH
                rayDirY = 0/effHEIGHT + camDirY
                rayDirZ = camDirZ - camDirX * 0 / effWIDTH

                pointX = camPosX-camDirX
                pointY = camPosY-camDirY
                pointZ = camPosZ-camDirZ

                point = basic_cast([pointX, pointY, pointZ], [rayDirX, rayDirY, rayDirZ])
                pointX, pointY, pointZ = point

                if point == [-1,-1,-1]:
                    continue

                #t = 2*int(time.perf_counter() - holdingDown)
                t = 2*int(holdingDown)
                if t != 0:
                    for i in range(-t,t):
                        for j in range(-t,t):
                            for k in range(-t,t):
                                world.set_voxel(pointX+i, pointY+j, pointZ+k, "air")
                else:
                    world.set_voxel(pointX, pointY, pointZ, "air")
        iteration += 1
        #print(count)
        print(GET_COUNT)
        print(LAMBDA_COUNT)
        buffer = []
