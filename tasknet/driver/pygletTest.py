import pyglet

window = pyglet.window.Window()
batch = pyglet.graphics.Batch()
circle = pyglet.shapes.Circle(x=100, y=150, radius=100, color=(50, 225, 30),batch=batch)

@window.event
def on_draw():
    window.clear()
    batch.draw()


a = pyglet.app.run()
circle.x = 10
