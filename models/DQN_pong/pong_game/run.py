import pygame

WIDTH = 500
HEIGHT = 500

bricksize = (10, 40)
ball_radius = 7
padding = 30
padding_right = WIDTH - (padding + bricksize[0])


p1pos = HEIGHT/2. - bricksize[1]/2

p2pos = p1pos

ballpos = (int(HEIGHT/2), int(WIDTH/2))



def updateDraw(surf):
    #draw p1 paddle
    pygame.draw.rect(surf, (255, 255, 255), (padding, p1pos, 10, 50))

    #draw p2 paddle
    pygame.draw.rect(surf, (255, 255, 255), (padding_right, p2pos, 10, 50))

    pygame.draw.circle(surf, (255, 255, 255), ballpos, ball_radius)



def main():
    print('starting game')
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT))
    isRunning = True
    while isRunning:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isRunning = False
        updateDraw(DISPLAYSURF)
        pygame.display.update()

    pygame.quit()





if __name__=="__main__":
    # call the main function
    main()