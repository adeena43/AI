class HospitalEnvironment:
    def __init__(self):
        self.rooms = {
            101: {"medicine": "A", "schedule": "9:00 AM", "needs_attention": False},
            102: {"medicine": "B", "schedule": "10:00 AM", "needs_attention": True},
            103: {"medicine": "C", "schedule": "9:30 AM", "needs_attention": False},

        }
        self.medicine_storage = ["A", "B", "C", "D"] 
        self.robot_position = "storage" 

    def display_status(self):
        print(f"Robot position: {self.robot_position}")
        print(f"Rooms status: {self.rooms}")

class DeliveryRobot:
    def __init__(self, environment):
        self.environment = environment
        self.picked_medicine = None
        self.delivered = False

    def move_to(self, location):
        print(f"Moving to {location}...")
        self.environment.robot_position = location

    def pick_medicine(self, medicine):
        if medicine in self.environment.medicine_storage:
            self.picked_medicine = medicine
            self.environment.medicine_storage.remove(medicine)
            print(f"Picked up medicine: {medicine}")
        else:
            print(f"Medicine {medicine} not available in storage.")

    def deliver_medicine(self, room):
        if room in self.environment.rooms:
            room_info = self.environment.rooms[room]
            if self.picked_medicine == room_info["medicine"]:
                print(f"Delivering {self.picked_medicine} to room {room}.")
                self.delivered = True
                room_info["medicine"] = None 
            else:
                print(f"Incorrect medicine for room {room}. Need {room_info['medicine']}.")

    def scan_patient_id(self, room):
        if self.delivered:
            print(f"Scanning patient ID in room {room}.")
        else:
            print("Medicine delivery not completed. Cannot scan patient ID.")

    def alert_staff(self, room):
        if self.environment.rooms[room]["needs_attention"]:
            print(f"Alerting staff for assistance in room {room}.")
        else:
            print(f"No immediate attention needed in room {room}.")

    def perform_delivery_task(self, room):
        print(f"Starting delivery to room {room}.")
        room_info = self.environment.rooms[room]
        self.move_to("storage")
        self.pick_medicine(room_info["medicine"])
        self.move_to(room)
        self.deliver_medicine(room)
        self.scan_patient_id(room)
        self.alert_staff(room)
environment = HospitalEnvironment()
robot = DeliveryRobot(environment)
robot.perform_delivery_task(101)  
robot.perform_delivery_task(102)  
