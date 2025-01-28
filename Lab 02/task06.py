class Environment:
    def __init__(self):
        self.rooms = [
            ['a', 'b', 'c'],
            ['d', 'e', 'f'],
            ['g', 'h', 'i']
        ]
        self.status = [
            ['safe', 'safe', 'safe'],
            ['safe', 'fire', 'safe'],
            ['safe', 'safe', 'fire']
        ]

    def show(self):
        print("Rooms:")
        for row in self.rooms:
            print(row)
        print("\nRoom Status:")
        for row in self.status:
            print(row)


class Robot:
    def __init__(self):
        pass

    def process(self, room_layout, room_status):
        for i in range(len(room_layout)):
            for j in range(len(room_layout[i])):
                print(f"\nRobot moving to room {room_layout[i][j]}...")
                if room_status[i][j] == 'fire':
                    print(f"Fire detected in room {room_layout[i][j]}! Extinguishing fire...")
                    room_status[i][j] = 'safe'
                else:
                    print(f"Room {room_layout[i][j]} is safe.")
                print("Updated Status:")
                self.show(room_layout, room_status)

    def show(self, room_layout, room_status):
        for i in range(len(room_layout)):
            for j in range(len(room_layout[i])):
                if room_status[i][j] == 'fire':
                    print("🔥", end=" ")
                else:
                    print("|", end=" ")
            print()

env = Environment()
robot = Robot()

print("Initial State:")
env.show()

robot.process(env.rooms, env.status)

print("\nFinal State:")
env.show()
