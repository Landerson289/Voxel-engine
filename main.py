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

### Bugs
# Lines in the water

import pygame
import time
import random
import math as maths
import octree

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


COLOUR_COUNT = 0

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
    "colour" : lambda in_colour, point, dir, rayLength, colour : colour,
    "blend" : lambda in_colour, point, dir, rayLength, colour, coefficient : blend(colour, in_colour, coefficient),
    "random" : lambda in_colour, point, dir, rayLength : (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
    "fog_blend": lambda in_colour, point, dir, rayLength, colour : blend(colour, in_colour, rayLength/sqrt3),
    "texture": lambda in_colour, point, dir, rayLength, texture : get_pixel_from_texture(texture, point),
    "blend_texture" : lambda in_colour, point, dir, rayLength, texture, coefficient : blend(get_pixel_from_texture(texture, point), in_colour, coefficient),
    "light_responsive_colour" : lambda in_colour, point, dir, rayLength, colour : brightnessAdjustment(in_colour, colour),
    "light_responsive_texture" : lambda in_colour, point, dir, rayLength, texture : brightnessAdjustment(in_colour, get_pixel_from_texture(texture, point))
}

blocks = { # Name : [Direction_effect, Colour_effect, direction_arguments, colour_arguments]
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
    "water" : ["specular_reflection", "blend", [], [(0,0,255), 0.1]],#[(0,0,255), 0.1]]
    "lightResponsiveRed" : ["diffuse_reflection", "light_responsive_colour", [], [(255,0,0)]],
    "lightResponsiveGrass" : ["diffuse_reflection", "light_responsive_texture", [], ["grass"]],
    "lightResponsiveWood" : ["diffuse_reflection", "light_responsive_texture", [], ["wood"]],
    "lightResponsiveDirt" : ["diffuse_reflection", "light_responsive_texture", [], ["dirt"]],
    #"Double reflector"
}


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
        s = time.perf_counter()
        self.generate_world()
        e = time.perf_counter()
        print("create array", e-s)

        s = time.perf_counter()
        self.octree = octree.Octree(self.voxel_array, 0, 0, 0, len(self.voxel_array))
        e = time.perf_counter()
        print("create octree", e-s)
        
        del self.voxel_array
    
    def generate_world(self):
        self.voxel_array = []
        for i in range(128):
            self.voxel_array.append([])
            for j in range(128):
                self.voxel_array[-1].append([])
                for k in range(128):
                    
                    #colour = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    #condition = j > 32

                    #colour = (0,random.randint(150,255),0) if j > 5 else (random.randint(150,255),random.randint(150,255),random.randint(150,255))
                    #condition = j > 10+5*maths.sin(i/10)+5*maths.sin(k/10)
                    condition = j > 10 + 5*sins[i] + 5*sins[k]

                    #colour = random.choice(list(textureCoords.keys()))#"grass"
                    if j < 20:
                        #colour = "grass"
                        colour = "lightResponsiveGrass"
                    elif j < 50:
                        colour = "lightResponsiveDirt"
                    else:
                        colour = "lightResponsiveStone"

                    #colour = "grass"

                    if condition == 0 and j > 13:
                        condition = 1
                        colour = "water"

                    self.voxel_array[-1][-1].append(colour if condition else "air")
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
        
        for i in range(128):
            x = random.randrange(0,128)
            z = random.randrange(12,128)
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
        

        for i in range(2048):
            v = 1
            while v != "air":
                x = random.randrange(0, 128)
                y = random.randrange(0, 5)
                z = random.randrange(0, 128)
                #self.set_voxel(x,y,z,"mist")
                v = self.voxel_array[int(x)][int(y)][int(z)]
            self.voxel_array[int(x)][int(y)][int(z)] = "mist"

        #for x in range(128):
        #    for z in range(128):
        #        self.voxel_array[int(x)][0][int(z)] = "mirror"


        self.voxel_array[9][5][20] = "whiteLight"
        

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

def cast_ray(start, rayDir, depth=1):
    global COLOUR_COUNT
    #global count
    #count += 1

    if depth > 200:
        return (255,255,0)

    voxelName, currentTree = world.get_voxel(*start)
    voxel = blocks[voxelName]
    
    COLOUR_COUNT += 1
    newRayDir = direction_functions[voxel[0]](start, rayDir, *voxel[2])
    #newRayDir = normalise(newRayDir)

    if newRayDir == [0,0,0]:
        COLOUR_COUNT += 1
        return colour_functions[voxel[1]](None, start, newRayDir, None, *voxel[3])
    
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
    COLOUR_COUNT += 1
    out_colour = colour_functions[voxel[1]](in_colour, start, newRayDir, DL, *voxel[3])
    return out_colour
    #return colour_functions[voxel[1]](cast_ray(end, newRayDir, depth+1), start, newRayDir, DL, *voxel[3])
      

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

    camPosX,camPosY,camPosZ = [10,10,10]
    #camDirX,camDirY,camDirZ = [0,0,1]

    yaw = 0#maths.pi/2

    camDirX,camDirY,camDirZ = [maths.sin(yaw),0,maths.cos(yaw)]

    running = True

    holdingDown = False

    iteration = 0

    print("WORLD GENERATED", time.perf_counter() - st)

    while running:
        st = time.perf_counter()
        for imageX in range(-effWIDTH//2, effWIDTH//2):
            for imageY in range(-effHEIGHT//2, effHEIGHT//2):
                pointX = camPosX-camDirX
                pointY = camPosY-camDirY
                pointZ = camPosZ-camDirZ

                rayDirX = camDirX + camDirZ * imageX / effWIDTH
                rayDirY = imageY/effHEIGHT + camDirY
                rayDirZ = camDirZ - camDirX * imageX / effWIDTH

                colour = cast_ray([pointX, pointY, pointZ], [rayDirX, rayDirY, rayDirZ])

                if iteration == 0:
                    current_colour = colour
                else:
                    current_colour = screen.get_at((RESOLUTION*(imageX+effWIDTH//2), RESOLUTION*(imageY+effHEIGHT//2)))
                
                averageColour = [(iteration*current_colour[i] + colour[i])/(iteration+1) for i in range(3)]

                #pygame.draw.rect(screen, colour, (RESOLUTION*(imageX+effWIDTH//2), RESOLUTION*(imageY+effHEIGHT//2), RESOLUTION, RESOLUTION))
                screen.set_at((RESOLUTION*(imageX+effWIDTH//2), RESOLUTION*(imageY+effHEIGHT//2)), averageColour)

            pygame.draw.circle(screen, (50,50,50), (WIDTH//2, HEIGHT//2),5)
            pygame.display.update()
        
        et = time.perf_counter()
        print(et-st)

        vel = [0,0,0]

        for event in pygame.event.get():
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

                camPosZ += (camDirZ * vel[2] - camDirX * vel[0])*0.5
                camPosY += vel[1]*0.5
                camPosX += (camDirZ * vel[0] + camDirX * vel[2])*0.5
            elif event.type == pygame.MOUSEBUTTONDOWN:
                holdingDown = time.perf_counter()

            elif event.type == pygame.MOUSEBUTTONUP:
                rayDirX = camDirX + camDirZ * 0 / effWIDTH
                rayDirY = 0/effHEIGHT + camDirY
                rayDirZ = camDirZ - camDirX * 0 / effWIDTH

                pointX = camPosX-camDirX
                pointY = camPosY-camDirY
                pointZ = camPosZ-camDirZ

                #pointX, pointY, pointZ, voxel, flags = cast_ray(rayDirX, rayDirY, rayDirZ, pointX, pointY, pointZ, "")

                #t = 2*int(time.perf_counter() - holdingDown)
                #if t != 0:
                #    for i in range(-t,t):
                #        for j in range(-t,t):
                #            for k in range(-t,t):
                #                world.set_voxel(pointX+i, pointY+j, pointZ+k, 0 if "2" not in flags else "dirt")
                #else:
                #    world.set_voxel(pointX, pointY, pointZ, 0 if "2" not in flags else "dirt")
        iteration += 1
        #print(count)
        print(COLOUR_COUNT)
