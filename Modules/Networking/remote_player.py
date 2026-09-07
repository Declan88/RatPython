from panda3d.core import NodePath


class RemotePlayer:
    def __init__(self, app, steam_id):
        self.app = app
        self.steam_id = steam_id

        # Visual representation for other connected peers in the lobby
        self.node = self.app.loader.loadModel("models/smiley")
        self.node.reparentTo(self.app.render)
        self.node.setScale(1.0)

    def update_transform(self, pos, hpr):
        self.node.setPos(*pos)
        self.node.setHpr(*hpr)

    def destroy(self):
        self.node.removeNode()
