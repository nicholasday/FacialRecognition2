class User():
    signins = []

    def __init__(self, name, id, pizza, team):
        self.name = name
        self.id = id
        self.pizza = pizza
        self.team = team

    def __repr__(self):
        return f'User({self.name}, {self.id}, {self.pizza}, {self.team})'