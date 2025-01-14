import pygame

def got_quit_event(events):
    for event in events:
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.unicode == 'q':
                return True
    return False

def display_text(screen,font,color,background,Text,pos):
    left,top = pos
    for line in Text:
        size = font.size(line)
        ren = font.render(line, 0, color, background)
        screen.blit(ren, (left,top))
        top += int( round(size[1] * 1.2) )
    return (left,top)
